//! Seeded serial graph sampling for the modern Euclidean layout.

use std::fmt;

use crate::fuzzy_graph::CsrGraph;
use crate::layout::{LayoutError, validate_csr_inputs};
use crate::modern_layout::{ModernLayout, ModernLayoutConfig, ModernLayoutError, ModernOptimizer};
use crate::rng::{Mt19937, draw_negative_seed};

use super::{EdgeSamplingCursor, SamplingError, seeded_train_positive_interval};

/// Graph sampling parameters for the seeded serial modern layout.
#[derive(Clone, Copy, Debug)]
pub struct ModernSamplingConfig {
    /// Requested negative sample rate. Zero disables repulsive samples.
    pub negative_sample_rate: usize,
    /// Exclude the source vertex and its retained graph neighbors from sampling.
    pub exclude_graph_neighbors: bool,
    /// Use a projected vertex order when the vertex count exceeds this value.
    pub negative_selection_range: usize,
}

impl Default for ModernSamplingConfig {
    fn default() -> Self {
        Self {
            negative_sample_rate: 5,
            exclude_graph_neighbors: false,
            negative_selection_range: 200_000,
        }
    }
}

/// Samples consumed by an epoch and its resulting coordinates.
#[derive(Debug)]
pub struct ModernEpoch<'a> {
    /// Directed attractions in retained CSR order.
    pub positive: &'a [(usize, usize)],
    /// Directed repulsions in dispatch order, including repeated targets.
    pub negative: &'a [(usize, usize)],
    /// Row-major coordinates after the epoch.
    pub coordinates: &'a [f32],
}

/// A rejected graph, initialization, or epoch.
#[derive(Clone, Debug, PartialEq)]
pub enum ModernSamplingError {
    InvalidVertexCount(usize),
    Graph(LayoutError),
    Layout(ModernLayoutError),
    Schedule(SamplingError),
    SampleCapacity,
    /// The projection is degenerate or two keys are within rounding error of
    /// one another. The wheel's order for this case is unresolved.
    AmbiguousProjection {
        epoch: usize,
    },
}

impl fmt::Display for ModernSamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVertexCount(count) => write!(formatter, "invalid vertex count: {count}"),
            Self::Graph(error) => error.fmt(formatter),
            Self::Layout(error) => error.fmt(formatter),
            Self::Schedule(error) => error.fmt(formatter),
            Self::SampleCapacity => write!(formatter, "cannot store the epoch's negative samples"),
            Self::AmbiguousProjection { epoch } => {
                write!(formatter, "ambiguous negative projection at epoch {epoch}")
            }
        }
    }
}

impl std::error::Error for ModernSamplingError {}

/// Modern Euclidean layout with native seeded sample dispatch.
///
/// Pass the initialized coordinates and a caller RNG positioned after any
/// initialization draws. The input CSR stays intact. Its weak memberships are
/// omitted from the schedule and the graph-neighbor exclusion lists.
pub struct ModernSampledLayout {
    layout: ModernLayout,
    n_epochs: usize,
    n_components: usize,
    sampling: ModernSamplingConfig,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    cursors: Vec<EdgeSamplingCursor>,
    pending_cursors: Vec<EdgeSamplingCursor>,
    shuffled: Vec<usize>,
    adam_shuffle: Option<Vec<usize>>,
    projection: Vec<ProjectionKey>,
    direction: Vec<f64>,
    ambiguous_projection: bool,
    positive: Vec<(usize, usize)>,
    negative: Vec<(usize, usize)>,
}

impl ModernSampledLayout {
    /// Seed the dispatch from the caller RNG even when the epoch count is zero.
    /// Validation completes before the caller RNG advances.
    pub fn new(
        graph: &CsrGraph,
        coordinates: Vec<f32>,
        layout: ModernLayoutConfig,
        sampling: ModernSamplingConfig,
        caller_rng: &mut Mt19937,
    ) -> Result<Self, ModernSamplingError> {
        let n_rows = graph.n_rows;
        if n_rows == 0 || n_rows > i32::MAX as usize {
            return Err(ModernSamplingError::InvalidVertexCount(n_rows));
        }
        if graph.indptr.last().copied() != Some(graph.indices.len()) {
            return Err(ModernSamplingError::Graph(LayoutError::InvalidGraphShape {
                n_rows,
                n_cols: graph.n_cols,
                indptr: graph.indptr.len(),
                indices: graph.indices.len(),
                data: graph.data.len(),
            }));
        }
        let n_components = layout.n_components;
        let n_epochs = layout.n_epochs;
        let adam_shuffle = (layout.optimizer == ModernOptimizer::Adam
            && n_rows <= sampling.negative_selection_range)
            .then(|| (0..n_rows).collect());
        let layout_state =
            ModernLayout::new(coordinates, layout).map_err(ModernSamplingError::Layout)?;
        validate_csr_inputs(
            layout_state.coordinates(),
            n_components,
            graph,
            layout.a,
            layout.b,
            layout.learning_rate,
        )
        .map_err(ModernSamplingError::Graph)?;

        let max_weight = graph.data.iter().copied().fold(0.0_f32, f32::max);
        let cutoff_epochs = if n_epochs > 10 {
            n_epochs
        } else if n_rows <= 10_000 {
            500
        } else {
            200
        };
        let cutoff = max_weight / cutoff_epochs as f32;
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::new();
        let mut cursors = Vec::new();
        indptr.push(0);
        for head in 0..n_rows {
            for edge in graph.indptr[head]..graph.indptr[head + 1] {
                let weight = graph.data[edge];
                if weight < cutoff || weight == 0.0 {
                    continue;
                }
                if let Some(interval) = seeded_train_positive_interval(n_epochs, max_weight, weight)
                {
                    let mut cursor = EdgeSamplingCursor::from_positive_interval(
                        interval,
                        sampling.negative_sample_rate.max(1) as f64,
                    )
                    .map_err(ModernSamplingError::Schedule)?;
                    // The multiplication follows the division. Rewriting it
                    // as a scaled rate changes when some backlogs truncate.
                    cursor.negative_interval *= 1.5;
                    cursor.next_negative = cursor.negative_interval;
                    cursors.push(cursor);
                    indices.push(graph.indices[edge]);
                }
            }
            indptr.push(indices.len());
        }

        // The public caller's RNG consumes the seed triplet before any shuffle.
        draw_negative_seed(|| caller_rng.next_u32());
        Ok(Self {
            layout: layout_state,
            n_epochs,
            n_components,
            sampling,
            indptr,
            indices,
            pending_cursors: cursors.clone(),
            cursors,
            shuffled: (0..n_rows).collect(),
            adam_shuffle,
            projection: Vec::new(),
            direction: Vec::new(),
            ambiguous_projection: false,
            positive: Vec::new(),
            negative: Vec::new(),
        })
    }

    pub fn coordinates(&self) -> &[f32] {
        self.layout.coordinates()
    }

    pub fn into_coordinates(self) -> Vec<f32> {
        self.layout.into_coordinates()
    }

    pub fn next_epoch(&self) -> usize {
        self.layout.next_epoch()
    }

    /// Dispatch and apply one epoch, then prepare the next candidate order.
    ///
    /// Keep the same RNG and its Gaussian cache from initialization to the
    /// final epoch. A rejected epoch preserves the layout, schedules, RNG and
    /// Gaussian cache. Self draws consume a slot and contribute no force.
    pub fn step(
        &mut self,
        caller_rng: &mut Mt19937,
        gaussian_cache: &mut Option<f64>,
    ) -> Result<ModernEpoch<'_>, ModernSamplingError> {
        let epoch = self.next_epoch();
        if epoch == self.n_epochs {
            return Err(ModernSamplingError::Layout(ModernLayoutError::EpochLimit));
        }
        self.positive.clear();
        self.negative.clear();
        self.pending_cursors.clone_from(&self.cursors);
        let n_rows = self.shuffled.len();
        let exclude = self.sampling.exclude_graph_neighbors;
        for (head, span) in self.indptr.windows(2).enumerate() {
            let neighbors = &self.indices[span[0]..span[1]];
            let all_excluded = exclude
                && neighbors.len() - usize::from(neighbors.binary_search(&head).is_ok())
                    == n_rows - 1;
            for (offset, &tail) in neighbors.iter().enumerate() {
                let edge = span[0] + offset;
                let cursor = &mut self.pending_cursors[edge];
                if !cursor
                    .positive_due(epoch as f64)
                    .map_err(ModernSamplingError::Schedule)?
                {
                    continue;
                }
                self.positive.push((head, tail));
                if self.sampling.negative_sample_rate == 0 || all_excluded {
                    continue;
                }
                let count = cursor
                    .take_negative_samples(epoch as f64)
                    .map_err(ModernSamplingError::Schedule)?;
                if count == 0 {
                    continue;
                }

                if self.ambiguous_projection {
                    return Err(ModernSamplingError::AmbiguousProjection { epoch });
                }
                self.negative
                    .try_reserve(count)
                    .map_err(|_| ModernSamplingError::SampleCapacity)?;
                let start = (edge as u128 * (epoch as u128 + 1) % n_rows as u128) as usize;
                let stride = if exclude { 1 } else { edge % n_rows };
                let mut position = start;
                let mut remaining = count;
                while remaining != 0 {
                    let target = if self.projection.is_empty() {
                        self.shuffled[position]
                    } else {
                        self.projection[position].vertex
                    };
                    position = (position + stride) % n_rows;
                    if exclude && (head == target || neighbors.binary_search(&target).is_ok()) {
                        continue;
                    }
                    remaining -= 1;
                    if head != target {
                        self.negative.push((head, target));
                    }
                }
            }
        }
        self.layout
            .step(&self.positive, &self.negative)
            .map_err(ModernSamplingError::Layout)?;
        std::mem::swap(&mut self.cursors, &mut self.pending_cursors);
        // Full-range Adam shuffles two independent orders each epoch. The
        // second order supplies the next epoch's negative candidates.
        if let Some(order) = &mut self.adam_shuffle {
            shuffle(order, caller_rng);
        }
        shuffle(&mut self.shuffled, caller_rng);
        if n_rows > self.sampling.negative_selection_range {
            self.ambiguous_projection = project_vertices(
                self.layout.coordinates(),
                self.n_components,
                caller_rng,
                gaussian_cache,
                &mut self.direction,
                &mut self.projection,
            );
        }
        Ok(ModernEpoch {
            positive: &self.positive,
            negative: &self.negative,
            coordinates: self.layout.coordinates(),
        })
    }
}

fn shuffle(vertices: &mut [usize], rng: &mut Mt19937) {
    for i in (1..vertices.len()).rev() {
        let mask = (i + 1).next_power_of_two() - 1;
        let j = loop {
            let candidate = rng.next_u32() as usize & mask;
            if candidate <= i {
                break candidate;
            }
        };
        vertices.swap(i, j);
    }
}

struct ProjectionKey {
    vertex: usize,
    key: f64,
    error: f64,
}

fn project_vertices(
    coordinates: &[f32],
    n_components: usize,
    rng: &mut Mt19937,
    gaussian_cache: &mut Option<f64>,
    direction: &mut Vec<f64>,
    projection: &mut Vec<ProjectionKey>,
) -> bool {
    direction.clear();
    direction.extend((0..n_components).map(|_| rng.next_standard_normal(gaussian_cache)));
    let norm = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return true;
    }
    for value in direction.iter_mut() {
        *value /= norm;
    }
    let roundoff = 4.0 * n_components as f64 * f64::EPSILON;
    projection.clear();
    for (vertex, row) in coordinates.chunks_exact(n_components).enumerate() {
        let mut key = 0.0;
        let mut magnitude = 0.0;
        for (&coordinate, &coefficient) in row.iter().zip(direction.iter()) {
            let product = f64::from(coordinate) * coefficient;
            key += product;
            magnitude += product.abs();
        }
        projection.push(ProjectionKey {
            vertex,
            key,
            // Includes the roundoff of both the direction normalization and
            // the dot product. BLAS can use a different accumulation order.
            error: magnitude * roundoff,
        });
    }
    projection.sort_unstable_by(|a, b| a.key.total_cmp(&b.key));
    projection
        .windows(2)
        .any(|pair| pair[1].key - pair[0].key <= pair[0].error + pair[1].error)
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::{Mt19937, draw_negative_seed, shuffle};

    #[derive(Deserialize)]
    struct Fixture {
        cases: Vec<Case>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        seed: u32,
        n_rows: usize,
        n_components: usize,
        negative_selection_range: usize,
        seed_draw: [i64; 4],
        random_epochs: Vec<RandomEvent>,
        adam_random_epochs: Option<Vec<RandomEvent>>,
    }

    #[derive(Deserialize)]
    struct RandomEvent {
        permutation: Vec<usize>,
        shuffle_cursor: usize,
        normal_bits: Option<Vec<u64>>,
        normal_cursor: Option<usize>,
    }

    #[test]
    fn modern_rng_matches_public_seed_shuffle_and_normal_draws() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../reference/fixtures/umap-learn-0.6.0-modern-sampling-ubuntu-x86_64.json"
        ))
        .unwrap();
        for case in fixture.cases {
            for (adam, events) in [
                (false, Some(case.random_epochs)),
                (true, case.adam_random_epochs),
            ] {
                let Some(events) = events else {
                    continue;
                };
                let mut rng = Mt19937::new(case.seed);
                let mut cache = None;
                let lanes = draw_negative_seed(|| rng.next_u32());
                assert_eq!(lanes, case.seed_draw[..3], "{}", case.name);
                assert_eq!(rng.state().1, case.seed_draw[3] as usize);
                let mut first: Vec<_> = (0..case.n_rows).collect();
                let mut second = first.clone();
                let two_shuffles = adam && case.n_rows <= case.negative_selection_range;
                for (index, event) in events.into_iter().enumerate() {
                    let actual = if two_shuffles && index % 2 == 0 {
                        &mut second
                    } else {
                        &mut first
                    };
                    shuffle(actual, &mut rng);
                    assert_eq!(*actual, event.permutation, "{} event {index}", case.name);
                    assert_eq!(rng.state().1, event.shuffle_cursor, "{}", case.name);
                    if let Some(normals) = event.normal_bits {
                        assert_eq!(normals.len(), case.n_components);
                        for expected in normals {
                            let actual = rng.next_standard_normal(&mut cache).to_bits();
                            assert!(
                                actual.abs_diff(expected) <= 2,
                                "{} event {index}",
                                case.name
                            );
                        }
                        assert_eq!(rng.state().1, event.normal_cursor.unwrap());
                    }
                }
            }
        }
    }
}

//! Dense Euclidean Adam and momentum epoch updates from supplied samples.
//!
//! This stage takes already initialized coordinates and the accepted samples
//! for each epoch. Graph construction, sample timing and selection, and
//! initialization rescaling are the caller's responsibility.

use std::fmt;

use crate::layout::DistanceReduction;

/// Update rule for a modern dense Euclidean layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModernOptimizer {
    /// Adaptive first and second moments.
    Adam,
    /// Learning-rate decay with a carried update.
    Momentum,
}

/// Settings for a modern layout starting from initialized coordinates.
#[derive(Clone, Copy, Debug)]
pub struct ModernLayoutConfig {
    /// Per-coordinate optimizer and learning-rate schedule.
    pub optimizer: ModernOptimizer,
    /// Number of coordinates in each row.
    pub n_components: usize,
    /// Total number of epochs used to determine the schedule.
    pub n_epochs: usize,
    /// Initial learning rate. Adam's late phase has a fixed schedule.
    pub learning_rate: f64,
    /// Scale of the low-dimensional membership curve.
    pub a: f64,
    /// Exponent of the low-dimensional membership curve.
    pub b: f64,
    /// Multiplier applied after limiting each negative sample's force.
    pub repulsion_strength: f64,
}

/// A rejected initialization or epoch. A rejected epoch leaves the layout
/// and its running state at the preceding epoch.
#[derive(Clone, Debug, PartialEq)]
pub enum ModernLayoutError {
    InvalidShape { values: usize, n_components: usize },
    InvalidParameter { name: &'static str, value: f64 },
    InvalidCoordinate { index: usize, value: f32 },
    InvalidRow { row: usize, n_rows: usize },
    NonFiniteArithmetic,
    EpochLimit,
}

impl fmt::Display for ModernLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape {
                values,
                n_components,
            } => write!(
                f,
                "cannot arrange {values} coordinates into rows of {n_components}"
            ),
            Self::InvalidParameter { name, value } => write!(f, "invalid {name}: {value}"),
            Self::InvalidCoordinate { index, value } => {
                write!(f, "coordinate {index} is not finite: {value}")
            }
            Self::InvalidRow { row, n_rows } => write!(f, "row {row} is outside 0..{n_rows}"),
            Self::NonFiniteArithmetic => write!(f, "non-finite modern layout update"),
            Self::EpochLimit => write!(f, "the configured number of epochs has already run"),
        }
    }
}

impl std::error::Error for ModernLayoutError {}

/// An in-progress modern layout with zero-initialized running state.
pub struct ModernLayout {
    config: ModernLayoutConfig,
    epoch: usize,
    coordinates: Vec<f32>,
    moments: Vec<[f32; 2]>,
    forces: Vec<f32>,
    pending: Vec<(f32, [f32; 2])>,
}

impl ModernLayout {
    /// Start at the supplied row-major coordinates. Ownership of the buffer
    /// transfers to the layout without a copy.
    pub fn new(
        coordinates: Vec<f32>,
        config: ModernLayoutConfig,
    ) -> Result<Self, ModernLayoutError> {
        let len = coordinates.len();
        if config.n_components == 0 || !len.is_multiple_of(config.n_components) {
            return Err(ModernLayoutError::InvalidShape {
                values: len,
                n_components: config.n_components,
            });
        }
        for (name, value, positive) in [
            ("learning_rate", config.learning_rate, false),
            ("a", config.a, true),
            ("b", config.b, true),
            ("repulsion_strength", config.repulsion_strength, false),
        ] {
            if !value.is_finite() || value < 0.0 || (positive && value == 0.0) {
                return Err(ModernLayoutError::InvalidParameter { name, value });
            }
        }
        for (index, &value) in coordinates.iter().enumerate() {
            if !value.is_finite() {
                return Err(ModernLayoutError::InvalidCoordinate { index, value });
            }
        }
        Ok(Self {
            config,
            epoch: 0,
            coordinates,
            moments: vec![[0.0; 2]; len],
            forces: vec![0.0; len],
            pending: vec![(0.0, [0.0; 2]); len],
        })
    }

    /// The current row-major coordinates.
    pub fn coordinates(&self) -> &[f32] {
        &self.coordinates
    }

    /// Return the row-major coordinates and consume the running state.
    pub fn into_coordinates(self) -> Vec<f32> {
        self.coordinates
    }

    /// The index of the next epoch to run, starting at zero.
    pub fn next_epoch(&self) -> usize {
        self.epoch
    }

    /// Accumulate the directed positive and negative samples and commit one
    /// epoch. Each `(source, target)` contributes only to its source. To
    /// attract both endpoints, supply both directions. Repeated samples
    /// contribute once per occurrence. Self samples and coincident endpoints
    /// contribute zero.
    ///
    /// All forces read the coordinates from the start of the epoch. The
    /// return value is the post-update snapshot for the epoch just run. To
    /// replay the observed unit-weight schedule, supply an empty epoch 0.
    pub fn step(
        &mut self,
        positive: &[(usize, usize)],
        negative: &[(usize, usize)],
    ) -> Result<&[f32], ModernLayoutError> {
        if self.epoch == self.config.n_epochs {
            return Err(ModernLayoutError::EpochLimit);
        }

        let components = self.config.n_components;
        let n_rows = self.coordinates.len() / components;
        self.forces.fill(0.0);
        for (samples, repulsive) in [(positive, false), (negative, true)] {
            for &(source, target) in samples {
                for row in [source, target] {
                    if row >= n_rows {
                        return Err(ModernLayoutError::InvalidRow { row, n_rows });
                    }
                }
                if source == target || (repulsive && self.config.repulsion_strength == 0.0) {
                    continue;
                }
                let source = source * components..(source + 1) * components;
                let target = target * components..(target + 1) * components;
                let head = &self.coordinates[source.clone()];
                let tail = &self.coordinates[target];
                let q = f64::from(DistanceReduction::Sequential.squared(head, tail));
                if !q.is_finite() {
                    return Err(ModernLayoutError::NonFiniteArithmetic);
                }
                if q == 0.0 {
                    continue;
                }

                let curve = 1.0 + self.config.a * q.powf(self.config.b);
                if !curve.is_finite() {
                    return Err(ModernLayoutError::NonFiniteArithmetic);
                }
                let scale = if repulsive {
                    let norm = q.sqrt();
                    let magnitude = (2.0 * self.config.b / ((0.001 + q) * curve)) * norm;
                    self.config.repulsion_strength * (2.0 * magnitude.tanh() / norm)
                } else {
                    -2.0 * self.config.a * self.config.b * q.powf(self.config.b - 1.0) / curve
                };
                for ((total, &head), &tail) in self.forces[source].iter_mut().zip(head).zip(tail) {
                    *total = (f64::from(*total) + scale * f64::from(head - tail)) as f32;
                }
            }
        }

        let update = EpochUpdate::new(self.config, self.epoch);
        for i in 0..self.coordinates.len() {
            let force = self.forces[i];
            let (coordinate, moments) = update.apply(self.coordinates[i], self.moments[i], force);
            if !force.is_finite()
                || !coordinate.is_finite()
                || !moments.iter().all(|v| v.is_finite())
            {
                return Err(ModernLayoutError::NonFiniteArithmetic);
            }
            self.pending[i] = (coordinate, moments);
        }

        for (i, &(coordinate, moments)) in self.pending.iter().enumerate() {
            self.coordinates[i] = coordinate;
            self.moments[i] = moments;
        }
        self.epoch += 1;
        Ok(&self.coordinates)
    }
}

const BETA1: f64 = 0.9_f32 as f64;
const BETA2: f64 = 0.99_f32 as f64;

enum EpochUpdate {
    Momentum { alpha: f64, carry: f64 },
    Adam { alpha: f64, bias1: f64, bias2: f64 },
}

impl EpochUpdate {
    fn new(config: ModernLayoutConfig, epoch: usize) -> Self {
        let epochs = config.n_epochs;
        match config.optimizer {
            ModernOptimizer::Momentum => {
                let remaining = 1.0 - epoch as f64 / epochs as f64;
                let previous = 1.0 - epoch.saturating_sub(1) as f64 / epochs as f64;
                Self::Momentum {
                    alpha: 0.125 * config.learning_rate * remaining * remaining,
                    carry: 0.5 - 0.125 * previous * previous,
                }
            }
            ModernOptimizer::Adam => {
                let early_epochs = (epochs / 2).min(100);
                let alpha = if epoch < early_epochs {
                    let remaining = 1.0 - epoch as f64 / early_epochs as f64;
                    0.1 + (2.0 * config.learning_rate - 0.1) * remaining * remaining
                } else {
                    0.2 - 0.15 * (epoch - early_epochs) as f64 / (epochs - early_epochs) as f64
                };
                let time = (epoch + 1) as f64;
                Self::Adam {
                    alpha,
                    bias1: 1.0 - BETA1.powf(time),
                    bias2: 1.0 - BETA2.powf(time),
                }
            }
        }
    }

    fn apply(&self, coordinate: f32, [first, second]: [f32; 2], force: f32) -> (f32, [f32; 2]) {
        match *self {
            Self::Momentum { alpha, carry } => {
                let update = (carry * f64::from(first) + alpha * f64::from(force)) as f32;
                (coordinate + update, [update, 0.0])
            }
            Self::Adam {
                alpha,
                bias1,
                bias2,
            } => {
                // A coordinate with no force leaves both of its moments intact.
                if force == 0.0 {
                    return (coordinate, [first, second]);
                }
                let first = (BETA1 * f64::from(first) + (1.0 - BETA1) * f64::from(force)) as f32;
                let second =
                    (BETA2 * f64::from(second) + (1.0 - BETA2) * f64::from(force * force)) as f32;
                let update = alpha * (f64::from(first) / bias1)
                    / ((f64::from(second) / bias2).sqrt() + 0.0002);
                ((f64::from(coordinate) + update) as f32, [first, second])
            }
        }
    }
}

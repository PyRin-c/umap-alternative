use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::fuzzy_graph::CsrGraph;
use umap_core::initialization::InitializationError;
use umap_core::modern_initialization::{
    ModernInit, ModernInitConfig, ModernInitError, ModernInitRescale, initialize_modern_layout,
};
use umap_core::rng::Mt19937;

const WHEEL: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-modern-init-ubuntu-x86_64.json");

#[derive(Deserialize)]
struct Fixture {
    graphs: BTreeMap<String, Graph>,
    cases: Vec<Case>,
    spectral: Vec<Case>,
    recursive: Vec<Recursive>,
}

#[derive(Deserialize)]
struct Graph {
    n_rows: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    weights_bits: Vec<u32>,
}

impl Graph {
    fn csr(&self) -> CsrGraph {
        CsrGraph {
            n_rows: self.n_rows,
            n_cols: self.n_rows,
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: floats(&self.weights_bits),
        }
    }
}

#[derive(Deserialize)]
struct Case {
    name: String,
    graph: String,
    method: String,
    factor: f64,
    n_components: usize,
    init: String,
    seed: u32,
    initial_bits: Vec<u32>,
    embedding_bits: Vec<u32>,
    rng_start: Option<Rng>,
    rng: Rng,
}

#[derive(Deserialize)]
struct Recursive {
    graph: String,
    seed: u32,
    threshold: usize,
    raw_bits: Vec<u32>,
    embedding_bits: Vec<u32>,
    auto_matches: bool,
}

#[derive(Deserialize)]
struct Rng {
    position: usize,
    gaussian: Option<f64>,
    next_words: Vec<u32>,
}

fn floats(bits: &[u32]) -> Vec<f32> {
    bits.iter().copied().map(f32::from_bits).collect()
}

fn replay(case: &Case, graph: &CsrGraph) -> Vec<f32> {
    let method = match case.method.as_str() {
        "graph_edge" => ModernInitRescale::GraphEdge,
        "per_axis" => ModernInitRescale::PerAxis,
        "isotropic" => ModernInitRescale::Isotropic,
        _ => unreachable!(),
    };
    let coordinates = floats(&case.initial_bits);
    let init = match case.init.as_str() {
        "explicit" => ModernInit::Explicit(&coordinates),
        "random" => ModernInit::Random,
        "spectral" => ModernInit::Spectral(&coordinates),
        _ => unreachable!(),
    };
    let config = ModernInitConfig {
        n_components: case.n_components,
        rescale: method,
        scale_factor: case.factor,
    };
    let mut rng = Mt19937::new(case.seed);
    let mut cache = None;
    if let Some(start) = &case.rng_start {
        for _ in 0..start.position {
            rng.next_u32();
        }
        cache = start.gaussian;
        let mut probe = rng.clone();
        let words: Vec<_> = (0..start.next_words.len())
            .map(|_| probe.next_u32())
            .collect();
        assert_eq!(words, start.next_words);
    }
    let actual = initialize_modern_layout(graph, init, config, &mut rng, &mut cache)
        .unwrap_or_else(|error| panic!("{}: {error}", case.name));
    // Public embedding calls draw three sampler seed words after initialization.
    for _ in 0..3 {
        assert!(rng.next_u32() < 0xffff_fffd);
    }
    assert_eq!(rng.state().1, case.rng.position, "{}", case.name);
    match (cache, case.rng.gaussian) {
        (None, None) => {}
        (Some(actual), Some(expected)) => {
            assert!(
                (actual - expected).abs() <= 4.0 * f64::EPSILON,
                "{}",
                case.name
            );
        }
        _ => panic!("{}: Gaussian cache presence differed", case.name),
    }
    let next: Vec<_> = (0..case.rng.next_words.len())
        .map(|_| rng.next_u32())
        .collect();
    assert_eq!(next, case.rng.next_words, "{}", case.name);
    actual
}

fn compare(name: &str, actual: &[f32], expected: &[u32]) -> (usize, f32) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    actual
        .iter()
        .zip(expected)
        .fold((0, 0.0_f32), |(count, max), (&a, &b)| {
            if a.to_bits() == b || (a.is_nan() && f32::from_bits(b).is_nan()) {
                (count, max)
            } else {
                assert!(
                    a.is_finite() && f32::from_bits(b).is_finite(),
                    "{name}: {a} vs {}",
                    f32::from_bits(b)
                );
                (count + 1, max.max((a - f32::from_bits(b)).abs()))
            }
        })
}

#[test]
fn fits_public_explicit_and_random_checkpoints() {
    let fixture: Fixture = serde_json::from_str(WHEEL).unwrap();
    for case in &fixture.cases {
        // Direct calls to the public embedding function use a separate robust target.
        if case.name.starts_with("direct-") {
            continue;
        }
        let actual = replay(case, &fixture.graphs[&case.graph].csr());
        let (different, max) = compare(&case.name, &actual, &case.embedding_bits);
        if case.method == "isotropic" {
            for (&value, &expected) in actual.iter().zip(&case.embedding_bits) {
                let expected = f32::from_bits(expected);
                // NumPy and Rust can round the logarithm and exponential differently.
                assert!(
                    (value - expected).abs() <= 1.5e-6 * expected.abs(),
                    "{}: {value} vs {expected} (max absolute difference {max:e})",
                    case.name,
                );
            }
        } else {
            assert_eq!(
                different, 0,
                "{} (max absolute difference {max:e})",
                case.name
            );
        }
    }
}

#[test]
fn spectral_and_recursive_coordinate_handoffs() {
    let fixture: Fixture = serde_json::from_str(WHEEL).unwrap();
    let graph = fixture.graphs["spectral"].csr();
    for case in &fixture.spectral {
        let actual = replay(case, &graph);
        let (_, max) = compare(&case.name, &actual, &case.embedding_bits);
        assert!(max <= 1e-5, "{}: {max:e}", case.name);
    }
    for case in &fixture.recursive {
        let graph = fixture.graphs[&case.graph].csr();
        assert!(case.auto_matches);
        let mut rng = Mt19937::new(case.seed);
        let start = rng.state();
        let config = ModernInitConfig {
            n_components: 2,
            rescale: ModernInitRescale::GraphEdge,
            scale_factor: 1.0,
        };
        let actual = initialize_modern_layout(
            &graph,
            ModernInit::Prepared(&floats(&case.raw_bits)),
            config,
            &mut rng,
            &mut None,
        )
        .unwrap();
        assert_eq!(start, rng.state());
        let (different, max) = compare("recursive handoff", &actual, &case.embedding_bits);
        assert_eq!(
            different, 0,
            "seed {} threshold {}: {max:e}",
            case.seed, case.threshold
        );
    }
}

#[test]
fn validates_stage_inputs_before_advancing_rng() {
    let fixture: Fixture = serde_json::from_str(WHEEL).unwrap();
    let mut graph = fixture.graphs["short-heavy"].csr();
    let mut rng = Mt19937::new(42);
    let start = rng.state();
    let mut config = ModernInitConfig {
        n_components: 2,
        rescale: ModernInitRescale::GraphEdge,
        scale_factor: 1.0,
    };
    for scale_factor in [f64::INFINITY, f64::NAN, 0.0, -1.0] {
        config.scale_factor = scale_factor;
        assert!(matches!(
            initialize_modern_layout(&graph, ModernInit::Random, config, &mut rng, &mut None),
            Err(ModernInitError::InvalidScale(_))
        ));
    }
    config.scale_factor = 1.0;
    config.n_components = 0;
    assert!(matches!(
        initialize_modern_layout(&graph, ModernInit::Random, config, &mut rng, &mut None),
        Err(ModernInitError::Coordinates(
            InitializationError::InvalidShape { .. }
        ))
    ));
    config.n_components = usize::MAX;
    assert!(matches!(
        initialize_modern_layout(&graph, ModernInit::Random, config, &mut rng, &mut None),
        Err(ModernInitError::Coordinates(
            InitializationError::ShapeOverflow { .. }
        ))
    ));
    config.n_components = 2;
    assert!(matches!(
        initialize_modern_layout(
            &graph,
            ModernInit::Explicit(&[0.0; 4]),
            config,
            &mut rng,
            &mut None
        ),
        Err(ModernInitError::Coordinates(
            InitializationError::InvalidShape { .. }
        ))
    ));
    let mut malformed = graph.clone();
    malformed.indptr[1] = malformed.indices.len() + 1;
    assert!(matches!(
        initialize_modern_layout(&malformed, ModernInit::Random, config, &mut rng, &mut None),
        Err(ModernInitError::Graph(_))
    ));
    malformed = graph.clone();
    malformed.indptr.fill(0);
    assert!(matches!(
        initialize_modern_layout(&malformed, ModernInit::Random, config, &mut rng, &mut None),
        Err(ModernInitError::Graph(_))
    ));
    malformed = graph.clone();
    malformed.indices[0] = graph.n_rows;
    assert!(matches!(
        initialize_modern_layout(&malformed, ModernInit::Random, config, &mut rng, &mut None),
        Err(ModernInitError::Graph(_))
    ));
    malformed = graph.clone();
    malformed.data[0] = f32::from_bits(1.0f32.to_bits() + 1);
    assert!(matches!(
        initialize_modern_layout(&malformed, ModernInit::Random, config, &mut rng, &mut None),
        Err(ModernInitError::Graph(_))
    ));
    assert!(matches!(
        initialize_modern_layout(
            &graph,
            ModernInit::Spectral(&[0.0; 8]),
            config,
            &mut rng,
            &mut None,
        ),
        Err(ModernInitError::ZeroSpectralSpan)
    ));
    assert!(matches!(
        initialize_modern_layout(
            &graph,
            ModernInit::Explicit(&[f32::NAN; 8]),
            config,
            &mut rng,
            &mut None
        ),
        Err(ModernInitError::Coordinates(
            InitializationError::NonFiniteValue { .. }
        ))
    ));
    graph.indices.clear();
    graph.data.clear();
    graph.indptr.fill(0);
    assert_eq!(
        initialize_modern_layout(&graph, ModernInit::Random, config, &mut rng, &mut None),
        Err(ModernInitError::EmptyGraph)
    );
    assert_eq!(rng.state(), start);
}

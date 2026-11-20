use serde::Deserialize;
use std::collections::BTreeMap;
use umap_core::inverse::{
    DenseInverseError, DenseInverseInput, DenseInverseLayout, InverseTriangulation,
    dense_inverse_epochs, fit_dense_inverse_scales, initialize_dense_inverse,
    inverse_transform_dense,
};

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
    tie_diagnostics: Vec<Case>,
    default_schedule: Vec<DefaultSchedule>,
}

#[derive(Deserialize)]
struct DefaultSchedule {
    query_rows: usize,
    first_row_bits: BTreeMap<String, Vec<u32>>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    n_features: usize,
    n_components: usize,
    n_neighbors: usize,
    a: f64,
    b: f64,
    initial_alpha: f64,
    transform_seed: u32,
    training: Vec<f32>,
    embedding: Vec<f32>,
    queries: Vec<f32>,
    simplices: Vec<usize>,
    locations: Vec<Option<usize>>,
    checkpoints: Vec<Checkpoint>,
}

impl Case {
    fn input(&self) -> DenseInverseInput<'_> {
        DenseInverseInput {
            training: &self.training,
            embedding: &self.embedding,
            queries: &self.queries,
            n_features: self.n_features,
            n_components: self.n_components,
            triangulation: InverseTriangulation {
                simplices: &self.simplices,
                query_simplices: &self.locations,
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Epochs {
    Scalar(usize),
    List(Vec<usize>),
}

impl Epochs {
    fn maximum(&self) -> Option<usize> {
        match self {
            Self::Scalar(epoch) => Some(*epoch),
            Self::List(epochs) => epochs.iter().copied().max(),
        }
    }
}

#[derive(Deserialize)]
struct Checkpoint {
    fit_epochs: Option<Epochs>,
    negative_sample_rate: f64,
    repulsion_strength: f64,
    bits: Vec<u32>,
    warnings: Vec<String>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!(
        "../../../reference/fixtures/umap-learn-0.6.0-inverse-windows-x86_64.json"
    ))
    .unwrap()
}

#[test]
fn replays_public_inverse_transform_with_fixed_fitted_inputs() {
    let mut exact_initial = 0;
    let mut total_initial = 0;
    let mut exact_attraction = 0;
    let mut total_attraction = 0;
    let mut exact_repulsion = 0;
    let mut total_repulsion = 0;
    for case in fixture().cases {
        let input = case.input();
        let scales =
            fit_dense_inverse_scales(&case.training, case.n_features, case.n_neighbors).unwrap();
        for check in &case.checkpoints {
            let fit_epochs = check.fit_epochs.as_ref().and_then(Epochs::maximum);
            let epochs = dense_inverse_epochs(fit_epochs, case.locations.len());
            let layout = DenseInverseLayout {
                n_epochs: epochs,
                initial_alpha: case.initial_alpha,
                repulsion_strength: check.repulsion_strength,
                negative_sample_rate: check.negative_sample_rate,
                seed: case.transform_seed,
            };
            let result = inverse_transform_dense(&input, &scales, case.a, case.b, layout).unwrap();
            let exterior = result.exterior_queries;
            let mut expected_warnings = Vec::new();
            if case.name == "exterior_distance_overflow" {
                expected_warnings.push("overflow encountered in multiply".to_owned());
            }
            if exterior > 0 {
                expected_warnings.push(format!(
                    "{exterior} inverse-transform point(s) lie outside the embedding convex hull; using nearest-vertex extrapolation."
                ));
            }
            assert_eq!(check.warnings, expected_warnings, "{}", case.name);
            let output = result.coordinates;
            assert_eq!(output.len(), check.bits.len());
            let matched = matching_bits(&output, &check.bits);
            eprintln!(
                "{} fit {fit_epochs:?} inverse {epochs} rate {} repulsion {} bits {matched}/{}",
                case.name,
                layout.negative_sample_rate,
                layout.repulsion_strength,
                output.len()
            );
            assert_eq!(
                matched,
                output.len(),
                "{} fit {fit_epochs:?} at rate {} with repulsion {}",
                case.name,
                layout.negative_sample_rate,
                layout.repulsion_strength
            );
            if epochs <= 1 {
                assert_eq!(
                    initialize_dense_inverse(&input, case.a, case.b)
                        .unwrap()
                        .coordinates,
                    output
                );
                exact_initial += matched;
                total_initial += output.len();
            } else if layout.repulsion_strength == 0.0 {
                exact_attraction += matched;
                total_attraction += output.len();
            } else {
                exact_repulsion += matched;
                total_repulsion += output.len();
            }
        }
    }
    eprintln!(
        "initialization {exact_initial}/{total_initial}, attraction {exact_attraction}/{total_attraction}, repulsion {exact_repulsion}/{total_repulsion}"
    );
}

fn matching_bits(output: &[f32], expected: &[u32]) -> usize {
    output
        .iter()
        .zip(expected)
        .filter(|(actual, bits)| actual.to_bits() == **bits)
        .count()
}

#[test]
#[ignore = "reports tie-selection differences on larger symmetric candidate sets"]
fn reports_inverse_distance_tie_bits() {
    let mut matched = 0;
    let mut total = 0;
    for case in fixture().tie_diagnostics {
        let output = initialize_dense_inverse(&case.input(), case.a, case.b)
            .unwrap()
            .coordinates;
        let expected = &case.checkpoints[0].bits;
        assert_eq!(output.len(), expected.len());
        let exact = matching_bits(&output, expected);
        eprintln!("{} {exact}/{}", case.name, expected.len());
        matched += exact;
        total += expected.len();
    }
    eprintln!("distance ties {matched}/{total}");
}

#[test]
fn rejects_invalid_matrices_and_tessellations() {
    let case = fixture().cases.swap_remove(0);
    let check = |input: DenseInverseInput<'_>, error| {
        assert_eq!(initialize_dense_inverse(&input, case.a, case.b), Err(error),);
    };
    check(
        DenseInverseInput {
            queries: &[],
            ..case.input()
        },
        DenseInverseError::Queries,
    );
    check(
        DenseInverseInput {
            queries: &[0.0],
            ..case.input()
        },
        DenseInverseError::Queries,
    );
    check(
        DenseInverseInput {
            queries: &[0.0, f32::NAN],
            ..case.input()
        },
        DenseInverseError::Queries,
    );
    check(
        DenseInverseInput {
            training: &case.training[..case.n_features],
            embedding: &case.embedding[..case.n_components],
            ..case.input()
        },
        DenseInverseError::Triangulation,
    );
    check(
        DenseInverseInput {
            triangulation: InverseTriangulation {
                simplices: &[0, 1, usize::MAX],
                query_simplices: &[Some(0)],
            },
            queries: &[0.0, 0.0],
            ..case.input()
        },
        DenseInverseError::Triangulation,
    );
}

#[test]
fn inverse_epoch_count_matches_public_boundaries() {
    assert_eq!(dense_inverse_epochs(Some(5), 10), 1);
    assert_eq!(dense_inverse_epochs(Some(6), 10), 2);
    assert_eq!(dense_inverse_epochs(Some(45), 10), 15);
    for case in fixture().default_schedule {
        let epochs = dense_inverse_epochs(None, case.query_rows);
        let equivalent = (epochs * 3).to_string();
        assert_eq!(
            case.first_row_bits["None"], case.first_row_bits[&equivalent],
            "query count {}",
            case.query_rows
        );
        assert_ne!(case.first_row_bits["90"], case.first_row_bits["300"]);
    }
}

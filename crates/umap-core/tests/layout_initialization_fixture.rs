//! Cross-check the deterministic initialization boundary against public fixtures.
//!
//! The layout fixtures record several `fit_transform` calls, but this test only
//! claims compatibility for the observed `n_epochs=0` checkpoints.  Later
//! optimization epochs are intentionally left to the graph/layout tests.

use serde::Deserialize;
use umap_core::{initialization, rng::Mt19937};

const UBUNTU_FIXTURE_JSON: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-layout-ubuntu-x86_64.json");
const WINDOWS_FIXTURE_JSON: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-layout-windows-x86_64.json");

#[derive(Debug, Deserialize)]
struct LayoutFixture {
    parameters: Parameters,
    cases: Cases,
    numpy_randomstate_fixture: RandomStateFixture,
}

#[derive(Debug, Deserialize)]
struct Parameters {
    n_components: usize,
    random_state: u32,
}

#[derive(Debug, Deserialize)]
struct Cases {
    random_init: RandomInitializationCase,
    explicit_init: ExplicitInitializationCase,
}

#[derive(Debug, Deserialize)]
struct RandomInitializationCase {
    init: String,
    seed: u32,
    checkpoints: Vec<Checkpoint>,
    n_epochs_0_matches_public_randomstate_normalization: bool,
}

#[derive(Debug, Deserialize)]
struct ExplicitInitializationCase {
    init: ExplicitInput,
    normalization_control: NormalizationControl,
    checkpoints: Vec<Checkpoint>,
    n_epochs_0_matches_column_normalization: bool,
}

#[derive(Debug, Deserialize)]
struct ExplicitInput {
    dtype: String,
    shape: Vec<usize>,
    values: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct NormalizationControl {
    expected: Float32Matrix,
}

#[derive(Debug, Deserialize)]
struct Checkpoint {
    n_epochs: u32,
    output: Float32Matrix,
}

#[derive(Debug, Deserialize)]
struct Float32Matrix {
    dtype: String,
    shape: Vec<usize>,
    bits_uint32: Vec<Vec<u32>>,
}

#[derive(Debug, Deserialize)]
struct RandomStateFixture {
    randint_public_call: RawUint32Call,
    random_sample_public_call: Float32Vector,
}

#[derive(Debug, Deserialize)]
struct RawUint32Call {
    dtype: String,
    raw_uint32: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct Float32Vector {
    dtype: String,
    shape: Vec<usize>,
    bits_uint32: Vec<u32>,
}

fn load_fixture(json: &str) -> LayoutFixture {
    serde_json::from_str(json).expect("public layout fixture must be valid JSON")
}

fn n_epochs_zero(checkpoints: &[Checkpoint]) -> &Float32Matrix {
    let matches = checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.n_epochs == 0)
        .collect::<Vec<_>>();
    assert_eq!(
        matches.len(),
        1,
        "fixture must contain one n_epochs=0 checkpoint"
    );
    &matches[0].output
}

fn assert_matrix_shape_and_bits(values: &[f32], expected: &Float32Matrix, context: &str) {
    assert_eq!(expected.dtype, "float32", "{context}: unexpected dtype");
    let [n_rows, n_components] = expected.shape.as_slice() else {
        panic!(
            "{context}: expected a two-dimensional shape, got {:?}",
            expected.shape
        );
    };
    assert_eq!(
        expected.bits_uint32.len(),
        *n_rows,
        "{context}: bit row count"
    );
    assert!(
        expected
            .bits_uint32
            .iter()
            .all(|row| row.len() == *n_components),
        "{context}: every bit row must have {n_components} columns"
    );
    assert_eq!(
        values.len(),
        n_rows * n_components,
        "{context}: value count"
    );

    for (index, value) in values.iter().enumerate() {
        let row = index / n_components;
        let column = index % n_components;
        let expected_bits = expected.bits_uint32[row][column];
        assert_eq!(
            value.to_bits(),
            expected_bits,
            "{context}: mismatch at row {row}, column {column}"
        );
    }
}

fn assert_raw_uint32_matches_public_randomstate(fixture: &LayoutFixture, context: &str) {
    let expected = &fixture.numpy_randomstate_fixture.randint_public_call;
    assert_eq!(expected.dtype, "uint32", "{context}: randint dtype");

    let mut generator = Mt19937::new(fixture.parameters.random_state);
    let actual = (0..expected.raw_uint32.len())
        .map(|_| generator.next_u32())
        .collect::<Vec<_>>();
    assert_eq!(actual, expected.raw_uint32, "{context}: raw MT19937 words");
}

fn assert_random_sample_matches_public_randomstate(fixture: &LayoutFixture, context: &str) {
    let expected = &fixture.numpy_randomstate_fixture.random_sample_public_call;
    assert_eq!(expected.dtype, "float32", "{context}: random_sample dtype");
    assert_eq!(expected.shape, vec![expected.bits_uint32.len()]);

    let mut generator = Mt19937::new(fixture.parameters.random_state);
    let actual = (0..expected.bits_uint32.len())
        .map(|_| (generator.next_f64() as f32).to_bits())
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected.bits_uint32,
        "{context}: random_sample f32 bits"
    );
}

fn assert_random_initialization_matches_n_epochs_zero(fixture: &LayoutFixture, context: &str) {
    let case = &fixture.cases.random_init;
    assert_eq!(case.init, "random", "{context}: initialization mode");
    assert_eq!(
        case.seed, fixture.parameters.random_state,
        "{context}: seed"
    );
    assert!(
        case.n_epochs_0_matches_public_randomstate_normalization,
        "{context}: fixture does not claim this n_epochs=0 case is observed"
    );

    let expected = n_epochs_zero(&case.checkpoints);
    let [n_rows, n_components] = expected.shape.as_slice() else {
        panic!("{context}: random checkpoint must be two-dimensional");
    };
    assert_eq!(*n_components, fixture.parameters.n_components);

    let actual = initialization::random_initialization(case.seed, *n_rows, *n_components)
        .expect("observed random initialization must be valid");
    assert_matrix_shape_and_bits(&actual, expected, context);
}

fn assert_explicit_initialization_matches_n_epochs_zero(fixture: &LayoutFixture, context: &str) {
    let case = &fixture.cases.explicit_init;
    assert!(
        case.n_epochs_0_matches_column_normalization,
        "{context}: fixture does not claim this n_epochs=0 case is observed"
    );

    assert_eq!(
        case.init.dtype, "float32",
        "{context}: explicit input dtype"
    );
    let [n_rows, n_components] = case.init.shape.as_slice() else {
        panic!("{context}: explicit input must be two-dimensional");
    };
    assert_eq!(*n_components, fixture.parameters.n_components);
    assert_eq!(
        case.init.values.len(),
        *n_rows,
        "{context}: input row count"
    );
    assert!(
        case.init
            .values
            .iter()
            .all(|row| row.len() == *n_components),
        "{context}: every explicit input row must have {n_components} columns"
    );

    let mut actual = case
        .init
        .values
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    initialization::normalize_columns(&mut actual, *n_rows, *n_components)
        .expect("observed explicit initialization must be valid");

    let expected = n_epochs_zero(&case.checkpoints);
    assert_matrix_shape_and_bits(&actual, expected, context);
    assert_matrix_shape_and_bits(
        &actual,
        &case.normalization_control.expected,
        &format!("{context}: normalization control"),
    );
}

#[test]
fn raw_uint32_matches_both_public_randomstate_fixtures() {
    for (context, json) in [
        ("Ubuntu", UBUNTU_FIXTURE_JSON),
        ("Windows", WINDOWS_FIXTURE_JSON),
    ] {
        let fixture = load_fixture(json);
        assert_raw_uint32_matches_public_randomstate(&fixture, context);
    }
}

#[test]
fn random_sample_float32_bits_match_both_public_randomstate_fixtures() {
    for (context, json) in [
        ("Ubuntu", UBUNTU_FIXTURE_JSON),
        ("Windows", WINDOWS_FIXTURE_JSON),
    ] {
        let fixture = load_fixture(json);
        assert_random_sample_matches_public_randomstate(&fixture, context);
    }
}

#[test]
fn random_initialization_matches_observed_n_epochs_zero_checkpoints() {
    for (context, json) in [
        ("Ubuntu", UBUNTU_FIXTURE_JSON),
        ("Windows", WINDOWS_FIXTURE_JSON),
    ] {
        let fixture = load_fixture(json);
        assert_random_initialization_matches_n_epochs_zero(&fixture, context);
    }
}

#[test]
fn explicit_initialization_matches_observed_n_epochs_zero_checkpoints() {
    for (context, json) in [
        ("Ubuntu", UBUNTU_FIXTURE_JSON),
        ("Windows", WINDOWS_FIXTURE_JSON),
    ] {
        let fixture = load_fixture(json);
        assert_explicit_initialization_matches_n_epochs_zero(&fixture, context);
    }
}

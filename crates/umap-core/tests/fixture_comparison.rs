//! Exact-bit comparison helpers for the pinned synthetic reference fixture.
//!
//! These tests exercise only fixture parsing and comparison. They do not run a
//! UMAP implementation and therefore do not establish compatibility with
//! `umap-learn`.

use serde::Deserialize;

const FIXTURE_JSON: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-windows-x86_64.json");

#[derive(Debug, Deserialize)]
struct Fixture {
    expected: Expected,
}

#[derive(Debug, Deserialize)]
struct Expected {
    dtype: String,
    shape: Vec<usize>,
    bits_uint32: Vec<Vec<u32>>,
}

fn load_fixture() -> Fixture {
    serde_json::from_str(FIXTURE_JSON).expect("pinned UMAP fixture must be valid JSON")
}

fn embedding_from_expected(expected: &Expected) -> Vec<Vec<f32>> {
    expected
        .bits_uint32
        .iter()
        .map(|row| row.iter().map(|bits| f32::from_bits(*bits)).collect())
        .collect()
}

/// Compare a candidate embedding against the expected shape and exact f32 bits.
///
/// Nested row/column indexing makes row and element order part of the
/// comparison. The error includes the first mismatching location and both
/// `f32::to_bits` values, which keeps failures useful even for very close
/// floating-point values.
fn compare_embedding(candidate: &[Vec<f32>], expected: &Expected) -> Result<(), String> {
    if expected.dtype != "float32" {
        return Err(format!(
            "fixture dtype mismatch: expected float32, got {:?}",
            expected.dtype
        ));
    }

    let [expected_rows, expected_columns] = expected.shape.as_slice() else {
        return Err(format!(
            "fixture shape must have two dimensions, got {:?}",
            expected.shape
        ));
    };

    if candidate.len() != *expected_rows {
        return Err(format!(
            "shape mismatch: expected ({expected_rows}, {expected_columns}), got {} rows",
            candidate.len()
        ));
    }

    if expected.bits_uint32.len() != *expected_rows {
        return Err(format!(
            "fixture bits row count mismatch: shape declares {expected_rows}, bits contain {}",
            expected.bits_uint32.len()
        ));
    }

    for (row_index, row) in candidate.iter().enumerate() {
        if row.len() != *expected_columns {
            return Err(format!(
                "shape mismatch at row {row_index}: expected {expected_columns} columns, got {}",
                row.len()
            ));
        }

        let expected_row = &expected.bits_uint32[row_index];
        if expected_row.len() != *expected_columns {
            return Err(format!(
                "fixture bits column count mismatch at row {row_index}: shape declares {expected_columns}, bits contain {}",
                expected_row.len()
            ));
        }

        for (column_index, actual) in row.iter().enumerate() {
            let expected_bits = expected_row[column_index];
            let actual_bits = actual.to_bits();
            if actual_bits != expected_bits {
                let expected_value = f32::from_bits(expected_bits);
                return Err(format!(
                    concat!(
                        "embedding mismatch at row {}, column {}: ",
                        "expected f32 bits 0x{:08x} (value {:?}), ",
                        "got f32 bits 0x{:08x} (value {:?})"
                    ),
                    row_index, column_index, expected_bits, expected_value, actual_bits, actual
                ));
            }
        }
    }

    Ok(())
}

#[test]
fn parser_reads_pinned_fixture_metadata() {
    let fixture = load_fixture();

    assert_eq!(fixture.expected.dtype, "float32");
    assert_eq!(fixture.expected.shape, vec![8, 2]);
    assert_eq!(fixture.expected.bits_uint32.len(), 8);
    assert!(
        fixture
            .expected
            .bits_uint32
            .iter()
            .all(|row| row.len() == 2)
    );
}

#[test]
fn comparator_accepts_exact_fixture_bits() {
    let fixture = load_fixture();
    let candidate = embedding_from_expected(&fixture.expected);

    compare_embedding(&candidate, &fixture.expected)
        .expect("the fixture converted back to f32 must compare exactly");
}

#[test]
fn comparator_rejects_one_bit_change_with_location() {
    let fixture = load_fixture();
    let mut candidate = embedding_from_expected(&fixture.expected);
    let row_index = 3;
    let column_index = 1;
    let expected_bits = candidate[row_index][column_index].to_bits();
    let actual_bits = expected_bits ^ 1;
    candidate[row_index][column_index] = f32::from_bits(actual_bits);

    let error = compare_embedding(&candidate, &fixture.expected)
        .expect_err("a one-bit change must fail exact comparison");

    assert!(
        error.contains("row 3, column 1"),
        "unexpected error: {error}"
    );
    assert!(
        error.contains(&format!("expected f32 bits 0x{expected_bits:08x}")),
        "unexpected error: {error}"
    );
    assert!(
        error.contains(&format!("bits 0x{actual_bits:08x}")),
        "unexpected error: {error}"
    );
}

#[test]
fn comparator_rejects_row_order_change() {
    let fixture = load_fixture();
    let mut candidate = embedding_from_expected(&fixture.expected);
    candidate.swap(0, 1);

    let error = compare_embedding(&candidate, &fixture.expected)
        .expect_err("reordering rows must fail exact comparison");

    assert!(
        error.contains("row 0, column 0"),
        "unexpected error: {error}"
    );
}

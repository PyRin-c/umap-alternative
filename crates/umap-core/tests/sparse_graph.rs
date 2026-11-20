use std::collections::BTreeMap;

use serde::Deserialize;
use umap_core::fuzzy_graph::{CsrGraph, GraphError};
use umap_core::neighbors::{DistanceMetric, Neighbor};
use umap_core::sparse::{
    SparseDistanceMetric, SparseGraphFitError, SparseKnnError, exhaustive_csr_knn, fit_sparse_graph,
};

use DistanceMetric::{Chebyshev, Euclidean, Manhattan};

#[test]
fn merges_shared_and_disjoint_columns_with_empty_rows() {
    let indptr = [0, 2, 6, 6];
    let indices = [0, 2, 1, 2, 3, usize::MAX - 1];
    let values = [3.0, 4.0, -4.0, 1.0, -0.0, 12.0];
    for (metric, ab, ac, bc) in [
        (Euclidean, 178.0_f32.sqrt(), 5.0, 161.0_f32.sqrt()),
        (Manhattan, 22.0, 7.0, 17.0),
        (Chebyshev, 12.0, 4.0, 12.0),
    ] {
        let neighbors = |k| {
            exhaustive_csr_knn(&indptr, &indices, &values, 3, usize::MAX, k, metric)
                .expect("valid CSR feature matrix")
        };
        let distances = [[0.0, ab, ac], [ab, 0.0, bc], [ac, bc, 0.0]];
        let middle_order = if metric == Chebyshev {
            [1, 0, 2]
        } else {
            [1, 2, 0]
        };
        let orders = [[0, 2, 1], middle_order, [2, 0, 1]];
        for k in 1..=3 {
            for (row, result) in neighbors(k).iter().enumerate() {
                let expected: Vec<_> = orders[row][..k]
                    .iter()
                    .map(|&index| Neighbor {
                        index,
                        distance: distances[row][index],
                    })
                    .collect();
                assert_eq!(*result, expected, "{metric:?}, row {row}, k {k}");
            }
        }
    }
}

#[test]
fn ranks_equal_zero_distances_by_candidate_row_index() {
    let indptr = [0, 1, 1, 2, 2];
    let indices = [1, 0];
    let values = [-0.0, 0.0];
    for metric in [Euclidean, Manhattan, Chebyshev] {
        for k in 1..=4 {
            let neighbors = exhaustive_csr_knn(&indptr, &indices, &values, 4, 2, k, metric)
                .expect("zero feature vectors are valid");
            assert_eq!(neighbors.len(), 4);
            for row in neighbors {
                assert_eq!(row.len(), k);
                for (index, neighbor) in row.iter().enumerate() {
                    assert_eq!(neighbor.index, index);
                    assert_eq!(neighbor.distance.to_bits(), 0.0_f32.to_bits());
                }
            }
        }
    }

    assert_eq!(
        fit_sparse_graph(&indptr, &indices, &values, 4, 2, 2, Euclidean),
        Err(SparseGraphFitError::Graph(GraphError::MissingSelf {
            row: 2
        }))
    );
    assert_eq!(
        fit_sparse_graph(&[0, 0, 0, 0, 0], &[], &[], 4, 2, 4, Euclidean),
        Ok(CsrGraph {
            n_rows: 4,
            n_cols: 4,
            indptr: vec![0, 3, 6, 9, 12],
            indices: vec![1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            data: vec![1.0; 12],
        })
    );
}

#[test]
fn fits_one_row_and_accepts_empty_feature_ranges() {
    let graph = CsrGraph {
        n_rows: 1,
        n_cols: 1,
        indptr: vec![0, 0],
        indices: vec![],
        data: vec![],
    };
    assert_eq!(
        fit_sparse_graph(&[0, 1], &[3], &[f32::MAX], 1, 4, 1, Euclidean),
        Ok(graph.clone())
    );
    assert_eq!(
        fit_sparse_graph(&[0, 0], &[], &[], 1, 0, 1, Euclidean),
        Ok(graph)
    );
}

#[test]
fn rejects_out_of_range_neighbor_counts() {
    for (indptr, n_rows, k) in [
        (&[0][..], 0, 0),
        (&[0], 0, 1),
        (&[0, 0], 1, 0),
        (&[0, 0], 1, 2),
    ] {
        let error = SparseKnnError::InvalidK { k, n_rows };
        assert_eq!(
            exhaustive_csr_knn(indptr, &[], &[], n_rows, 2, k, Euclidean),
            Err(error.clone())
        );
        assert_eq!(
            fit_sparse_graph(indptr, &[], &[], n_rows, 2, k, Euclidean),
            Err(SparseGraphFitError::Knn(error))
        );
    }
}

#[test]
fn rejects_invalid_pointer_and_entry_lengths() {
    for (indptr, indices, values, n_rows) in [
        (&[][..], &[][..], &[][..], 1),
        (&[0][..], &[][..], &[][..], 1),
        (&[0, 0, 0][..], &[][..], &[][..], 1),
        (&[0][..], &[][..], &[][..], usize::MAX),
        (&[0, 1][..], &[0][..], &[][..], 1),
        (&[0, 1][..], &[][..], &[0.0][..], 1),
    ] {
        assert_eq!(
            exhaustive_csr_knn(indptr, indices, values, n_rows, 1, 1, Euclidean),
            Err(SparseKnnError::InvalidShape {
                n_rows,
                n_features: 1,
                offsets: indptr.len(),
                indices: indices.len(),
                values: values.len(),
            })
        );
    }
}

#[test]
fn rejects_invalid_row_offsets_before_slicing() {
    for (first, last) in [(1, 1), (0, 0), (0, 2)] {
        assert_eq!(
            exhaustive_csr_knn(&[first, last], &[0], &[1.0], 1, 1, 1, Euclidean),
            Err(SparseKnnError::InvalidEndpoints {
                first,
                last,
                entries: 1,
            })
        );
    }
    for (indptr, row, start, end) in [
        ([0, 1, 0, 1], 1, 1, 0),
        ([0, usize::MAX, 1, 1], 0, 0, usize::MAX),
    ] {
        assert_eq!(
            exhaustive_csr_knn(&indptr, &[0], &[1.0], 3, 1, 1, Euclidean),
            Err(SparseKnnError::InvalidRowOffsets {
                row,
                start,
                end,
                entries: 1,
            })
        );
    }
}

#[test]
fn rejects_out_of_range_duplicate_and_unsorted_columns() {
    for (column, n_features) in [(2, 2), (0, 0), (usize::MAX, usize::MAX)] {
        assert_eq!(
            exhaustive_csr_knn(&[0, 0, 1], &[column], &[1.0], 2, n_features, 1, Euclidean),
            Err(SparseKnnError::InvalidColumnIndex {
                row: 1,
                position: 0,
                column,
                n_features,
            })
        );
    }
    for column in [0, 1] {
        assert_eq!(
            exhaustive_csr_knn(&[0, 2], &[1, column], &[0.0, 1.0], 1, 2, 1, Euclidean),
            Err(SparseKnnError::UnorderedColumnIndices {
                row: 0,
                position: 1,
                previous: 1,
                column,
            })
        );
    }
}

#[test]
fn rejects_nonfinite_values_and_reports_computed_overflow_at_fit() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let error = exhaustive_csr_knn(&[0, 1, 2], &[0, 0], &[0.0, value], 2, 1, 1, Euclidean)
            .expect_err("stored values must be finite");
        assert!(
            matches!(error, SparseKnnError::NonFiniteValue { position: 1, value: invalid }
            if invalid.to_bits() == value.to_bits())
        );
    }
    for metric in [Euclidean, Manhattan, Chebyshev] {
        let indptr = [0, 1, 2];
        let indices = [0, 0];
        let values = [f32::MAX, -f32::MAX];
        let neighbors = exhaustive_csr_knn(&indptr, &indices, &values, 2, 1, 2, metric)
            .expect("stored values are finite");
        assert_eq!(neighbors[0][1].distance, f32::INFINITY);
        assert_eq!(neighbors[1][1].distance, f32::INFINITY);
        assert_eq!(
            fit_sparse_graph(&indptr, &indices, &values, 2, 1, 2, metric),
            Err(SparseGraphFitError::Graph(GraphError::InvalidDistance {
                row: 0,
                position: 1,
                index: 1,
                distance: f32::INFINITY,
            }))
        );
        assert!(fit_sparse_graph(&indptr, &indices, &values, 2, 1, 1, metric).is_ok());
    }
}

const PUBLIC_FIXTURE: &str =
    include_str!("../../../reference/fixtures/umap-learn-0.6.0-sparse-graph-windows-x86_64.json");

#[derive(Deserialize)]
struct Fixture {
    base_parameters: Parameters,
    inputs: BTreeMap<String, CsrRecord>,
    cases: BTreeMap<String, Case>,
}

#[derive(Deserialize)]
struct Parameters {
    n_neighbors: usize,
}

#[derive(Deserialize)]
struct CsrRecord {
    shape: [usize; 2],
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data_bits_uint32: Vec<u32>,
}

#[derive(Deserialize)]
struct Case {
    input: String,
    variant: String,
    graph: Option<CsrRecord>,
}

#[test]
fn sparse_feature_graphs_match_public_support_and_weight_bits() {
    let fixture: Fixture = serde_json::from_str(PUBLIC_FIXTURE).expect("valid public fixture");
    for (name, metric) in [
        ("features_csr_euclidean", Euclidean.into()),
        ("features_csr_manhattan", Manhattan.into()),
        ("features_csr_chebyshev", Chebyshev.into()),
        ("holdout_csr_euclidean", Euclidean.into()),
        ("features_csr_cosine", SparseDistanceMetric::Cosine),
        ("features_split_csr_cosine", SparseDistanceMetric::Cosine),
        ("holdout_csr_cosine", SparseDistanceMetric::Cosine),
    ] {
        let case = &fixture.cases[name];
        let csr = &fixture.inputs[&case.input];
        let mut values: Vec<_> = csr
            .data_bits_uint32
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect();
        let mut indptr = csr.indptr.clone();
        let mut indices = csr.indices.clone();
        if case.variant == "split_csr" {
            indptr.iter_mut().for_each(|offset| *offset *= 2);
            indices = indices.iter().flat_map(|&index| [index; 2]).collect();
            values = values.iter().flat_map(|value| [value * 0.5; 2]).collect();
        }
        let actual = fit_sparse_graph(
            &indptr,
            &indices,
            &values,
            csr.shape[0],
            csr.shape[1],
            fixture.base_parameters.n_neighbors,
            metric,
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let expected = case.graph.as_ref().expect("case contains a complete graph");
        assert_eq!([actual.n_rows, actual.n_cols], expected.shape, "{name}");
        assert_eq!(actual.indptr, expected.indptr, "{name}");
        assert_eq!(actual.indices, expected.indices, "{name}");
        let bits: Vec<_> = actual.data.iter().map(|value| value.to_bits()).collect();
        assert_eq!(bits, expected.data_bits_uint32, "{name}");
    }
}

#[test]
fn sparse_cosine_distinguishes_zero_norm_and_opposite_rows() {
    let values = [3.0, 6.0, -6.0, -0.0, f32::from_bits(1), f32::MAX];
    let indptr = [0, 1, 2, 3, 3, 4, 5, 6];
    let neighbors = exhaustive_csr_knn(
        &indptr,
        &[0; 6],
        &values,
        7,
        1,
        7,
        SparseDistanceMetric::Cosine,
    )
    .expect("valid CSR features");

    for (row, neighbors) in neighbors.iter().enumerate() {
        for neighbor in neighbors {
            let distance: f32 = match (row, neighbor.index) {
                (left, right) if left == right => 0.0,
                (0, 1) | (1, 0) => 0.0,
                (0 | 1, 2) | (2, 0 | 1) => 2.0,
                _ => 1.0,
            };
            assert_eq!(neighbor.distance.to_bits(), distance.to_bits());
        }
    }
}

#[test]
fn rounds_sparse_euclidean_squares_before_taking_the_root() {
    let neighbors = exhaustive_csr_knn(
        &[0, 2, 4],
        &[0, 2, 0, 2],
        &[0.12, 0.03, 0.29, 0.18],
        2,
        3,
        2,
        Euclidean,
    )
    .expect("valid CSR features");
    assert_eq!(neighbors[0][1].distance.to_bits(), 0x3e68_2828);
    assert_eq!(neighbors[1][1].distance.to_bits(), 0x3e68_2828);

    let fixture: Fixture = serde_json::from_str(PUBLIC_FIXTURE).expect("valid public fixture");
    let csr = &fixture.inputs["features"];
    let data: Vec<_> = csr
        .data_bits_uint32
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    let neighbors = exhaustive_csr_knn(
        &csr.indptr,
        &csr.indices,
        &data,
        csr.shape[0],
        csr.shape[1],
        csr.shape[0],
        Euclidean,
    )
    .expect("valid CSR fixture");
    let witness = neighbors[3]
        .iter()
        .find(|neighbor| neighbor.index == 8)
        .unwrap();
    assert_eq!(witness.distance.to_bits(), 1082362016);
}

use serde::Deserialize;
use std::collections::BTreeMap;
use umap_core::fuzzy_graph::{CsrGraph, FuzzySetOperation, combine_fuzzy_graphs};

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    options: [FitOptions; 2],
    operands: [GraphBits; 2],
    outputs: BTreeMap<String, Output>,
}

#[derive(Deserialize)]
struct FitOptions {
    n_epochs: usize,
}

#[derive(Deserialize)]
struct Output {
    graph: GraphBits,
    operands_after: [GraphBits; 2],
}

#[derive(Deserialize)]
struct GraphBits(Vec<usize>, Vec<usize>, Vec<u32>);

impl GraphBits {
    fn into_csr(self) -> CsrGraph {
        CsrGraph {
            n_rows: self.0.len() - 1,
            n_cols: self.0.len() - 1,
            indptr: self.0,
            indices: self.1,
            data: self.2.into_iter().map(f32::from_bits).collect(),
        }
    }
}

#[test]
fn public_estimator_composition_graphs() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../reference/fixtures/umap-learn-0.6.0-composition-x86_64.json"
    ))
    .unwrap();
    for case in fixture.cases {
        let n_epochs = case.options[0].n_epochs.max(case.options[1].n_epochs);
        let [left, right] = case.operands.map(GraphBits::into_csr);
        for (label, output) in case.outputs {
            let operation = match label.as_str() {
                "union" => FuzzySetOperation::Union,
                "intersection" => FuzzySetOperation::Intersection,
                "difference" => FuzzySetOperation::Difference,
                _ => panic!("unknown operation {label}"),
            };
            let mut actual_left = left.clone();
            let combined = combine_fuzzy_graphs(&mut actual_left, &right, operation).unwrap();
            let context = format!("{}/{label}", case.name);
            let [expected_left, expected_right] = output.operands_after.map(GraphBits::into_csr);
            assert_eq!(actual_left, expected_left, "{context} left operand");
            assert_eq!(right, expected_right, "{context} right operand");
            assert_eq!(combined.indptr, output.graph.0, "{context} offsets");
            assert_eq!(combined.indices, output.graph.1, "{context} columns");
            // The public estimator's layout stage zeroes weak memberships for
            // the captured epoch counts, leaving the stored entries in place.
            let maximum = combined.data.iter().copied().fold(0.0, f32::max);
            let cutoff = maximum
                / if n_epochs > 10 {
                    n_epochs as f32
                } else {
                    500.0
                };
            let bits = combined
                .data
                .iter()
                .map(|&value| if value < cutoff { 0 } else { value.to_bits() })
                .collect::<Vec<_>>();
            assert_eq!(bits, output.graph.2, "{context} weights");
        }
    }
}

#[test]
fn composition_checks_both_shapes_and_canonical_storage() {
    let mut empty = GraphBits(vec![0, 0], vec![], vec![]).into_csr();
    let mut malformed = empty.clone();
    malformed.indices.push(0);
    malformed.data.push(0.5);
    assert!(combine_fuzzy_graphs(&mut empty, &malformed, FuzzySetOperation::Union).is_err());
    let mut other_shape = GraphBits(vec![0, 0, 0], vec![], vec![]).into_csr();
    assert!(combine_fuzzy_graphs(&mut other_shape, &empty, FuzzySetOperation::Union).is_err());
    assert_eq!(
        combine_fuzzy_graphs(&mut empty.clone(), &empty, FuzzySetOperation::Union).unwrap(),
        empty
    );
}

use super::{
    GraphError, GraphOptions, GraphOptionsError, batch_scale_floor, fuzzy_simplicial_set,
    fuzzy_simplicial_set_with_options, row_parameters, sorted_neighbor_distances,
};
use crate::neighbors::Neighbor;
use serde::Deserialize;

const FIXTURE: &str = include_str!("../../tests/fixtures/graph_options.json");

#[derive(Deserialize)]
struct Fixture {
    datasets: Vec<Dataset>,
}

#[derive(Deserialize)]
struct Dataset {
    name: String,
    neighbors: Vec<Vec<usize>>,
    distance_bits: Vec<Vec<u32>>,
    profiles: Vec<Profile>,
}

#[derive(Deserialize)]
struct Profile {
    connectivity: f64,
    rho_bits: Vec<u32>,
    sigma_bits: Vec<u32>,
    graphs: Vec<ExpectedGraph>,
}

#[derive(Deserialize)]
struct ExpectedGraph {
    mix: f64,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data_bits: Vec<u32>,
    linux_x86_64_data_bits: Option<Vec<u32>>,
}

impl Dataset {
    fn neighbor_rows(&self) -> Vec<Vec<Neighbor>> {
        self.neighbors
            .iter()
            .zip(&self.distance_bits)
            .map(|(row, bits)| {
                row.iter()
                    .zip(bits)
                    .map(|(&index, &bits)| Neighbor {
                        index,
                        distance: f32::from_bits(bits),
                    })
                    .collect()
            })
            .collect()
    }
}

#[test]
fn local_parameters_match_public_graph_probes() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid graph option fixture");
    let mut differences = Vec::new();
    for dataset in fixture.datasets {
        let neighbors = dataset.neighbor_rows();
        let distances = sorted_neighbor_distances(&neighbors);
        let batch_floor = batch_scale_floor(&distances);
        for profile in dataset.profiles {
            for (row, distances) in distances.chunks_exact(neighbors[0].len()).enumerate() {
                let (rho, sigma) = row_parameters(distances, profile.connectivity, batch_floor);
                let actual = (rho.to_bits(), sigma.to_bits());
                let expected = (profile.rho_bits[row], profile.sigma_bits[row]);
                if actual != expected {
                    differences.push((
                        dataset.name.clone(),
                        profile.connectivity,
                        row,
                        expected,
                        actual,
                    ));
                }
            }
        }
    }
    assert!(
        differences.is_empty(),
        "offset and scale bit differences: {differences:?}"
    );
}

#[test]
fn graph_support_and_weights_match_public_graph_probes() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid graph option fixture");
    let mut differences = Vec::new();
    for dataset in fixture.datasets {
        let mut neighbors = dataset.neighbor_rows();
        // Input order is immaterial, including ties between self and duplicates.
        for row in &mut neighbors {
            row.reverse();
        }
        for profile in dataset.profiles {
            for expected in profile.graphs {
                let options = GraphOptions::new(profile.connectivity, expected.mix)
                    .expect("valid fixture options");
                let graph = fuzzy_simplicial_set_with_options(&neighbors, options)
                    .expect("valid fixture neighbors");
                let context = (&dataset.name, profile.connectivity, expected.mix);
                assert_eq!(graph.n_rows, neighbors.len(), "{context:?}");
                assert_eq!(graph.n_cols, neighbors.len(), "{context:?}");
                assert_eq!(graph.indptr, expected.indptr, "{context:?}");
                assert_eq!(graph.indices, expected.indices, "{context:?}");
                let data_bits = if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
                    expected
                        .linux_x86_64_data_bits
                        .as_deref()
                        .unwrap_or(&expected.data_bits)
                } else {
                    &expected.data_bits
                };
                assert_eq!(graph.data.len(), data_bits.len(), "{context:?}");
                for (entry, (&actual, &bits)) in graph.data.iter().zip(data_bits).enumerate() {
                    if actual.to_bits() != bits {
                        differences.push((
                            dataset.name.clone(),
                            profile.connectivity,
                            expected.mix,
                            entry,
                            bits,
                            actual.to_bits(),
                        ));
                    }
                }
            }
        }
    }
    assert!(
        differences.is_empty(),
        "graph weight bit differences: {differences:?}"
    );
}

#[test]
fn option_ranges_are_checked_before_rounding() {
    assert_eq!(GraphOptions::new(1.0, 1.0), Ok(GraphOptions::default()));
    for connectivity in [0.0, -0.0, f64::from_bits(1), f64::MAX] {
        for mix in [0.0, -0.0, f64::from_bits(1), 1.0_f64.next_down(), 1.0] {
            assert!(GraphOptions::new(connectivity, mix).is_ok());
        }
    }
    for connectivity in [
        -f64::from_bits(1),
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert!(matches!(
            GraphOptions::new(connectivity, 1.0),
            Err(GraphOptionsError::InvalidLocalConnectivity { local_connectivity })
                if local_connectivity.to_bits() == connectivity.to_bits()
        ));
    }
    for mix in [
        -f64::from_bits(1),
        1.0_f64.next_up(),
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert!(matches!(
            GraphOptions::new(1.0, mix),
            Err(GraphOptionsError::InvalidSetOpMixRatio { set_op_mix_ratio })
                if set_op_mix_ratio.to_bits() == mix.to_bits()
        ));
    }
}

#[test]
fn empty_neighbors_share_default_validation() {
    let options = GraphOptions::new(0.5, 0.0).expect("valid options");
    assert_eq!(
        fuzzy_simplicial_set_with_options(&[], options),
        fuzzy_simplicial_set(&[])
    );
    assert_eq!(
        fuzzy_simplicial_set_with_options(&[vec![]], options),
        Err(GraphError::EmptyRow { row: 0 })
    );
}

#[test]
fn original_graph_retains_zeros_with_default_parameters() {
    let neighbors: Vec<Vec<Neighbor>> = (0..5)
        .map(|row| {
            (0..5)
                .map(|index| Neighbor {
                    index,
                    distance: match (row, index) {
                        (0, 4) | (4, 0) => 2.0,
                        _ if row == index => 0.0,
                        _ => 1.0,
                    },
                })
                .collect()
        })
        .collect();
    let original = fuzzy_simplicial_set(&neighbors).expect("valid neighbors");
    let graph = fuzzy_simplicial_set_with_options(&neighbors, GraphOptions::default())
        .expect("valid neighbors");
    assert_eq!(original.indptr, [0, 4, 8, 12, 16, 20]);
    assert_eq!(original.data[3], 0.0);
    assert_eq!(original.data[16], 0.0);
    assert_eq!(graph.indptr, [0, 3, 7, 11, 15, 18]);
    assert_eq!(graph.data, vec![1.0; 18]);
}

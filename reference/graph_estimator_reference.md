# Public graph-estimator reference

Generated exclusively through the public `umap-learn` wheel API; no source, private attributes, or tests are used. Each successful call records public CSR fields and float32 IEEE-754 data bits.

Reproduce with the pinned environment:

```powershell
uv run --project reference --locked python reference/generate_graph_estimator_reference.py --output reference/graph_estimator_reference.md --json-output reference/graph_estimator_reference.json
```

The complete machine-readable capture follows.

```json
{
  "base_parameters": {
    "compatibility_layout": true,
    "init": "random",
    "metric": "euclidean",
    "n_components": 2,
    "n_epochs": 30,
    "n_jobs": 1,
    "n_neighbors": 3,
    "random_state": 42,
    "transform_mode": "graph"
  },
  "environment": {
    "machine": "AMD64",
    "packages": {
      "llvmlite": "0.50.0",
      "numba": "0.68.1",
      "numpy": "2.5.5",
      "pynndescent": "0.7.0",
      "scikit-learn": "1.9.1",
      "scipy": "1.18.1",
      "umap-learn": "0.6.0"
    },
    "platform": "Windows",
    "python": "CPython 3.13.12"
  },
  "inputs": {
    "baseline_float32": {
      "dtype": "float32",
      "shape": [
        4,
        2
      ],
      "values": [
        [
          0.0,
          0.0
        ],
        [
          1.0,
          0.10000000149011612
        ],
        [
          0.20000000298023224,
          1.2999999523162842
        ],
        [
          1.7000000476837158,
          2.0999999046325684
        ]
      ]
    },
    "float64_same_values": {
      "dtype": "float64",
      "shape": [
        4,
        2
      ],
      "values": [
        [
          0.0,
          0.0
        ],
        [
          1.0,
          0.1
        ],
        [
          0.2,
          1.3
        ],
        [
          1.7,
          2.1
        ]
      ]
    },
    "nan_float32": {
      "dtype": "float32",
      "shape": [
        4,
        2
      ],
      "values": [
        [
          0.0,
          0.0
        ],
        [
          1.0,
          0.10000000149011612
        ],
        [
          0.20000000298023224,
          1.2999999523162842
        ],
        [
          NaN,
          2.0999999046325684
        ]
      ]
    },
    "one_sample_float32": {
      "dtype": "float32",
      "shape": [
        1,
        2
      ],
      "values": [
        [
          0.0,
          0.0
        ]
      ]
    },
    "two_samples_float32": {
      "dtype": "float32",
      "shape": [
        2,
        2
      ],
      "values": [
        [
          0.0,
          0.0
        ],
        [
          1.0,
          0.0
        ]
      ]
    }
  },
  "observations": {
    "baseline_fit": {
      "graph": {
        "data_bits_uint32": [
          1065353216,
          1065353216,
          1065353216,
          1062463265,
          1058390034,
          1065353216,
          1062463265,
          1065353216,
          1058390034,
          1065353216
        ],
        "dtype": "float32",
        "format": "csr",
        "indices": [
          1,
          2,
          0,
          2,
          3,
          0,
          1,
          3,
          1,
          2
        ],
        "indptr": [
          0,
          2,
          5,
          8,
          10
        ],
        "shape": [
          4,
          4
        ],
        "type": "csr_matrix"
      },
      "model_has_graph": true,
      "operation": "fit",
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "euclidean",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 3,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "return_is_estimator": true,
      "return_type": "UMAP",
      "status": "ok",
      "warnings": []
    },
    "baseline_fit_transform": {
      "graph_is_output": true,
      "model_has_graph": true,
      "operation": "fit_transform",
      "output": {
        "data_bits_uint32": [
          1065353216,
          1065353216,
          1065353216,
          1062463265,
          1058390034,
          1065353216,
          1062463265,
          1065353216,
          1058390034,
          1065353216
        ],
        "dtype": "float32",
        "format": "csr",
        "indices": [
          1,
          2,
          0,
          2,
          3,
          0,
          1,
          3,
          1,
          2
        ],
        "indptr": [
          0,
          2,
          5,
          8,
          10
        ],
        "shape": [
          4,
          4
        ],
        "type": "csr_matrix"
      },
      "output_is_graph": true,
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "euclidean",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 3,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "return_type": "csr_matrix",
      "status": "ok",
      "warnings": []
    },
    "float64_input": {
      "graph_is_output": true,
      "model_has_graph": true,
      "operation": "fit_transform",
      "output": {
        "data_bits_uint32": [
          1065353216,
          1065353216,
          1065353216,
          1062463265,
          1058390034,
          1065353216,
          1062463265,
          1065353216,
          1058390034,
          1065353216
        ],
        "dtype": "float32",
        "format": "csr",
        "indices": [
          1,
          2,
          0,
          2,
          3,
          0,
          1,
          3,
          1,
          2
        ],
        "indptr": [
          0,
          2,
          5,
          8,
          10
        ],
        "shape": [
          4,
          4
        ],
        "type": "csr_matrix"
      },
      "output_is_graph": true,
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "euclidean",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 3,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "return_type": "csr_matrix",
      "same_data_bits_as_float32_baseline": true,
      "status": "ok",
      "warnings": []
    },
    "invalid_metric": {
      "exception_class": "ValueError",
      "exception_message": "metric is neither callable nor a recognised string",
      "model_has_graph": false,
      "operation": "fit_transform",
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "definitely-not-a-metric",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 3,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "status": "error",
      "warnings": []
    },
    "n_neighbors_1": {
      "exception_class": "ValueError",
      "exception_message": "n_neighbors must be greater than 1",
      "model_has_graph": false,
      "operation": "fit_transform",
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "euclidean",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 1,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "status": "error",
      "warnings": []
    },
    "n_neighbors_larger_than_sample_count": {
      "graph_is_output": true,
      "model_has_graph": true,
      "operation": "fit_transform",
      "output": {
        "data_bits_uint32": [
          1065353216,
          1065353216,
          1065353216,
          1062463265,
          1058390034,
          1065353216,
          1062463265,
          1065353216,
          1058390034,
          1065353216
        ],
        "dtype": "float32",
        "format": "csr",
        "indices": [
          1,
          2,
          0,
          2,
          3,
          0,
          1,
          3,
          1,
          2
        ],
        "indptr": [
          0,
          2,
          5,
          8,
          10
        ],
        "shape": [
          4,
          4
        ],
        "type": "csr_matrix"
      },
      "output_is_graph": true,
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "euclidean",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 10,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "return_type": "csr_matrix",
      "status": "ok",
      "warnings": [
        {
          "class": "UserWarning",
          "message": "n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1"
        }
      ]
    },
    "nan_default_validation": {
      "exception_class": "ValueError",
      "exception_message": "Input contains NaN.",
      "model_has_graph": false,
      "operation": "fit_transform",
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "euclidean",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 3,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "status": "error",
      "warnings": []
    },
    "one_sample_fit": {
      "model_has_graph": false,
      "operation": "fit",
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "euclidean",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 3,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "return_is_estimator": true,
      "return_type": "UMAP",
      "status": "ok",
      "warnings": []
    },
    "one_sample_fit_transform": {
      "exception_class": "AttributeError",
      "exception_message": "'UMAP' object has no attribute 'graph_'",
      "model_has_graph": false,
      "operation": "fit_transform",
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "euclidean",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 3,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "status": "error",
      "warnings": []
    },
    "two_samples_fit": {
      "graph": {
        "data_bits_uint32": [],
        "dtype": "float32",
        "format": "csr",
        "indices": [],
        "indptr": [
          0,
          0,
          0
        ],
        "shape": [
          2,
          2
        ],
        "type": "csr_matrix"
      },
      "model_has_graph": true,
      "operation": "fit",
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "euclidean",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 3,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "return_is_estimator": true,
      "return_type": "UMAP",
      "status": "ok",
      "warnings": [
        {
          "class": "UserWarning",
          "message": "n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1"
        },
        {
          "class": "UserWarning",
          "message": "A large number of your vertices were disconnected from the manifold.\nDisconnection_distance = inf has removed 0 edges.\nIt has fully disconnected 2 vertices.\nYou might consider using find_disconnected_points() to find and remove these points from your data.\nUse umap.utils.disconnected_vertices() to identify them."
        }
      ]
    },
    "two_samples_fit_transform": {
      "graph_is_output": true,
      "model_has_graph": true,
      "operation": "fit_transform",
      "output": {
        "data_bits_uint32": [],
        "dtype": "float32",
        "format": "csr",
        "indices": [],
        "indptr": [
          0,
          0,
          0
        ],
        "shape": [
          2,
          2
        ],
        "type": "csr_matrix"
      },
      "output_is_graph": true,
      "parameters": {
        "compatibility_layout": true,
        "init": "random",
        "metric": "euclidean",
        "n_components": 2,
        "n_epochs": 30,
        "n_jobs": 1,
        "n_neighbors": 3,
        "random_state": 42,
        "transform_mode": "graph"
      },
      "return_type": "csr_matrix",
      "status": "ok",
      "warnings": [
        {
          "class": "UserWarning",
          "message": "n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1"
        },
        {
          "class": "UserWarning",
          "message": "A large number of your vertices were disconnected from the manifold.\nDisconnection_distance = inf has removed 0 edges.\nIt has fully disconnected 2 vertices.\nYou might consider using find_disconnected_points() to find and remove these points from your data.\nUse umap.utils.disconnected_vertices() to identify them."
        }
      ]
    }
  },
  "operation": "UMAP(transform_mode='graph') public estimator contract"
}
```

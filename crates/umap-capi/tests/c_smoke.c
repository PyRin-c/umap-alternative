/* Compile with a C11 compiler and the crate's include directory. */
#include "umap_capi.h"

int main(void) {
    const float points[] = {0.0f, 0.0f, 1.0f, 0.0f};
    umap_capi_csr_graph_t *graph = 0;
    umap_capi_status_t status = umap_capi_fit_dense_graph(
        points,
        sizeof(points) / sizeof(points[0]),
        2,
        2,
        2,
        UMAP_CAPI_METRIC_EUCLIDEAN,
        &graph);
    if (status != UMAP_CAPI_STATUS_OK || graph == 0) {
        return 1;
    }

    umap_capi_csr_view_t view = {0};
    status = umap_capi_csr_graph_view(graph, &view);
    if (status != UMAP_CAPI_STATUS_OK || view.n_rows != 2 ||
        view.n_cols != 2 || view.indptr_len != 3 ||
        view.indices_len != view.data_len) {
        (void)umap_capi_csr_graph_release(graph);
        return 2;
    }

    return umap_capi_csr_graph_release(graph) == UMAP_CAPI_STATUS_OK ? 0 : 3;
}

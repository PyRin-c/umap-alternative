#include "umap_capi.h"

#include <stdio.h>

int main(void) {
    const float points[] = {0.0f, 0.0f, 1.0f, 0.0f, 1.1f, 1.0f, 0.1f, 1.2f};
    umap_capi_csr_graph_t *graph = NULL;
    umap_capi_status_t status = umap_capi_fit_dense_graph(
        points, sizeof points / sizeof points[0],
        4, 2, 3, UMAP_CAPI_METRIC_EUCLIDEAN, &graph);
    if (status != UMAP_CAPI_STATUS_OK) {
        fprintf(stderr, "Graph fit failed: %lu\n", (unsigned long)status);
        return 1;
    }

    umap_capi_csr_view_t view = {0};
    status = umap_capi_csr_graph_view(graph, &view);
    if (status == UMAP_CAPI_STATUS_OK) {
        printf("%zu rows, %zu stored weights\n", view.n_rows, view.data_len);
    }
    umap_capi_status_t released = umap_capi_csr_graph_release(graph);
    return status != UMAP_CAPI_STATUS_OK || released != UMAP_CAPI_STATUS_OK;
}

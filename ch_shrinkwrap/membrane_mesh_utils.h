#ifndef _membrane_mesh_utils_h_
#define _membrane_mesh_utils_h_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VECTORSIZE 3
// Note this must match NEIGHBORSIZE in triangle_mesh.py
// [DB - can we export this constant from the module and then use that in triangle_mesh.py so that we don't need to define it in two places?]
#define NEIGHBORSIZE 20

typedef struct {
    float position[VECTORSIZE];
    float normal[VECTORSIZE];
    int32_t halfedge;
    int32_t valence;
    int32_t neighbors[NEIGHBORSIZE];
    int32_t component;
} vertex_t;

typedef struct {
    float position[VECTORSIZE];
} points_t;

static PyObject *calculate_pt_cnt_dist_2(PyObject *self, PyObject *args);

#ifdef __cplusplus
}
#endif

#endif /* _membrane_mesh_utils_h_ */
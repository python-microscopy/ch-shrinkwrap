#ifndef _membrane_mesh_utils_h_
#define _membrane_mesh_utils_h_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef _triangle_mesh_utils_h_

#define VECTORSIZE 3
// Note this must match NEIGHBORSIZE in triangle_mesh.py
// [DB - can we export this constant from the module and then use that in triangle_mesh.py so that we don't need to define it in two places?]
#define NEIGHBORSIZE 20

typedef struct halfedge_t {
    int32_t vertex;
    int32_t face;
    int32_t twin;
    int32_t next;
    int32_t prev;
    float length;
    int32_t component;
} halfedge_t;

typedef struct face_t {
    int32_t halfedge;
    float normal[VECTORSIZE];
    float area;
    int32_t component;
} face_t;

typedef struct face_d { //flat version of face_t
    int32_t halfedge;
    float normal0;
    float normal1;
    float normal2;
    float area;
    int32_t component;
} face_d;

typedef struct vertex_t{
    float position[VECTORSIZE];
    float normal[VECTORSIZE];
    int32_t halfedge;
    int32_t valence;
    int32_t neighbors[NEIGHBORSIZE];
    int32_t component;
    int32_t locally_manifold;
} vertex_t;

typedef struct vertex_d { //flat version of vertex_t
    float position0;
    float position1;
    float position2;
    float normal0;
    float normal1;
    float normal2;
    int32_t halfedge;
    int32_t valence;
    int32_t neighbor0;
    int32_t neighbor1;
    int32_t neighbor2;
    int32_t neighbor3;
    int32_t neighbor4;
    int32_t neighbor5;
    int32_t neighbor6;
    int32_t neighbor7;
    int32_t neighbor8;
    int32_t neighbor9;
    int32_t neighbor10;
    int32_t neighbor11;
    int32_t neighbor12;
    int32_t neighbor13;
    int32_t neighbor14;
    int32_t neighbor15;
    int32_t neighbor16;
    int32_t neighbor17;
    int32_t neighbor18;
    int32_t neighbor19;
    int32_t component;
    int32_t locally_manifold;
} vertex_d;

#endif
typedef struct {
    float position[VECTORSIZE];
} points_t;

static PyObject *calculate_pt_cnt_dist_2(PyObject *self, PyObject *args);

#ifdef __cplusplus
}
#endif

#endif /* _membrane_mesh_utils_h_ */
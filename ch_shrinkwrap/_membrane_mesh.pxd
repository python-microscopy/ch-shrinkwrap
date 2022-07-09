cimport numpy as np

from PYME.experimental._triangle_mesh cimport TriangleMesh, NEIGHBORSIZE, VECTORSIZE, face_d, face_t, vertex_d, vertex_t, halfedge_t 

cdef extern from 'membrane_mesh_utils.h':
    cdef struct points_t:
        float position[VECTORSIZE]


cdef class MembraneMesh(TriangleMesh):
    cdef public float kc
    cdef public float kg
    cdef public float a
    cdef public float c
    cdef public float c0
    cdef public float step_size
    cdef public float beta_1
    cdef public float beta_2
    cdef public float eps
    cdef public int max_iter
    cdef public int remesh_frequency
    cdef public int delaunay_remesh_frequency
    cdef public float delaunay_eps
    cdef public object _E
    cdef public object _pE
    cdef object _k_0
    cdef object _k_1
    cdef object _e_0
    cdef object _e_1
    cdef object _dH
    cdef object _dK
    cdef object _dE_neighbors
    cdef float * _ck_0
    cdef float * _ck_1
    cdef float * _ce_0
    cdef float * _ce_1
    cdef float * _cH
    cdef float * _cK
    cdef float * _cE
    cdef float * _cpE
    cdef float * _cdH
    cdef float * _cdK
    cdef float * _cdE_neighbors
    cdef public int search_k
    cdef public float search_rad
    cdef public float skip_prob
    cdef object _tree
    cdef public object cg
    cdef object _points
    cdef object _sigma

    cdef public float neck_threshold_low
    cdef public float neck_threshold_high
    cdef public int neck_first_iter

    cdef curvature_grad_c(self, float dN=*, float skip_prob=*)
    cdef curvature_grad(self, float dN=*, float skip_prob=*)

    cdef point_attraction_grad_kdtree(self, np.ndarray points, np.ndarray sigma, float w=*, int search_k=*)
    cdef _holepunch_insert_square(self, np.int32_t edge0, np.int32_t edge1, 
                    np.int32_t * new_edges,
                    np.int32_t * new_faces,
                    int n_edge_idx,
                    int n_face_idx)

    cdef grad(self, np.ndarray points, np.ndarray sigma)
        
    
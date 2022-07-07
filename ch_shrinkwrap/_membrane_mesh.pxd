cimport numpy as np

from PYME.experimental._triangle_mesh cimport TriangleMesh, NEIGHBORSIZE, VECTORSIZE, face_d, face_t, vertex_d, vertex_t, halfedge_t 

cdef extern from 'membrane_mesh_utils.h':
    cdef struct points_t:
        float position[VECTORSIZE]
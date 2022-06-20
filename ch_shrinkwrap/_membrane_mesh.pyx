cimport numpy as np
import numpy as np
import scipy.spatial
import cython
import math

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from PYME.experimental._triangle_mesh cimport TriangleMesh
from PYME.experimental._triangle_mesh import TriangleMesh
from PYME.experimental._triangle_mesh import VERTEX_DTYPE

from ch_shrinkwrap import membrane_mesh_utils
from ch_shrinkwrap import delaunay_utils

# Gradient descent methods
DESCENT_METHODS = ['conjugate_gradient', 'skeleton']
DEFAULT_DESCENT_METHOD = 'conjugate_gradient'

KBT = 0.0257  # eV # 4.11e-21  # joules
NM2M = 1
COS110 = -0.34202014332
PI = 3.1415927

MAX_VERTEX_COUNT = 2**31

I = np.eye(3, dtype=float)

USE_C = True

cdef extern from 'triangle_mesh_utils.h':
    const int NEIGHBORSIZE  # Note this must match NEIGHBORSIZE in triangle_mesh_utils.h
    const int VECTORSIZE

    cdef struct face_d:
        np.int32_t halfedge
        float normal0
        float normal1
        float normal2
        float area
        np.int32_t component

    cdef struct face_t:
        np.int32_t halfedge
        float normal[VECTORSIZE]
        float area
        np.int32_t component

    cdef struct vertex_d :
        float position0
        float position1
        float position2
        float normal0
        float normal1
        float normal2
        np.int32_t halfedge
        np.int32_t valence
        np.int32_t neighbor0
        np.int32_t neighbor1
        np.int32_t neighbor2
        np.int32_t neighbor3
        np.int32_t neighbor4
        np.int32_t neighbor5
        np.int32_t neighbor6
        np.int32_t neighbor7
        np.int32_t neighbor8
        np.int32_t neighbor9
        np.int32_t neighbor10
        np.int32_t neighbor11
        np.int32_t neighbor12
        np.int32_t neighbor13
        np.int32_t neighbor14
        np.int32_t neighbor15
        np.int32_t neighbor16
        np.int32_t neighbor17
        np.int32_t neighbor18
        np.int32_t neighbor19
        np.int32_t component
        np.int32_t locally_manifold
        
    cdef struct vertex_t:
        float position[VECTORSIZE]
        float normal[VECTORSIZE]
        np.int32_t halfedge
        np.int32_t valence
        np.int32_t neighbors[NEIGHBORSIZE]
        np.int32_t component
        np.int32_t locally_manifold

    cdef struct halfedge_t:
        np.int32_t vertex
        np.int32_t face
        np.int32_t twin
        np.int32_t next
        np.int32_t prev
        np.float32_t length
        np.int32_t component

cdef extern from 'membrane_mesh_utils.h':
    cdef struct points_t:
        float position[VECTORSIZE]

POINTS_DTYPE = np.dtype([('position', '3f4')])
POINTS_DTYPE2 = np.dtype([('position0', 'f4'), 
                          ('position1', 'f4'), 
                          ('position2', 'f4')])

cdef extern from "triangle_mesh_utils.c":
    void _update_face_normals(np.int32_t *f_idxs, halfedge_t *halfedges, vertex_t *vertices, face_t *faces, signed int n_idxs)
    void update_face_normal(int f_idx, halfedge_t *halfedges, vertex_d *vertices, face_d *faces)
    void update_single_vertex_neighbours(int v_idx, halfedge_t *halfedges, vertex_d *vertices, face_d *faces)

cdef extern from "membrane_mesh_utils.c":
    void fcompute_curvature_tensor_eig(float *Mvi, float *l1, float *l2, float *v1, float *v2) 
    void c_point_attraction_grad(points_t *attraction, points_t *points, float *sigma, void *vertices_, float w, float charge_sigma, int n_points, int n_vertices)
    void c_curvature_grad(void *vertices_, 
                         void *faces_,
                         halfedge_t *halfedges,
                         float dN,
                         float skip_prob,
                         int n_vertices,
                         float *k_0,
                         float *k_1,
                         float *e_0,
                         float *e_1,
                         float *H,
                         float *K,
                         float *dH,
                         float *dK,
                         float *E,
                         float *pE,
                         float *dE_neighbors,
                         float kc,
                         float kg,
                         float c0,
                         points_t *dEdN)

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

    def __init__(self, vertices=None, faces=None, mesh=None, **kwargs):
        TriangleMesh.__init__(self, vertices, faces, mesh, **kwargs)

        # Bending stiffness coefficients (in units of kbT)
        self.kc = 20.0*KBT  # eV
        self.kg = -20.0*KBT  # eV

        # Gradient weight
        self.a = 1.0
        self.c = 1.0

        # Spotaneous curvature
        # Keep in mind the curvature convention we're using. Into the surface is
        # "down" and positive curvature bends "up".
        self.c0 = 0.0 # -0.02  # nm^{-1} for a sphere of radius 50

        # Optimizer parameters
        self.step_size = 1
        self.beta_1 = 0.8
        self.beta_2 = 0.7
        self.eps = 1e-8
        self.max_iter = 250
        self.remesh_frequency = 100
        self.delaunay_remesh_frequency = 150

        # Coloring info
        #self._H = None
        #self._K = None
        #self._E = None
        #self._pE = None

        self._initialize_curvature_vectors()
        
        self.vertex_properties.extend(['E', 'curvature_principal0', 'curvature_principal1', 'point_dis', 'rms_point_sc']) #, 'puncture_candidates'])

        # Number of neighbors to use in self.point_attraction_grad_kdtree
        self.search_k = 200
        self.search_rad = 100

        # Percentage of vertices to skip on each refinement iteration
        self.skip_prob = 0.0

        # Pointcloud kdtree
        self._tree = None

        # self._puncture_test = False  # Toggle puncture testing
        # self._puncture_candidates = []

        for key, value in kwargs.items():
            setattr(self, key, value)

    # @property
    # def puncture_candidates(self):
    #     arr = np.zeros(self._vertices.shape[0])
    #     arr[self._puncture_candidates] = 1
    #     return arr

    @property
    def E(self):
        if not np.any(self._E):
            self._populate_curvature_grad()
            self._E[np.isnan(self._E)] = 0
        return self._E

    @property
    def pE(self):
        if not np.any(self._pE):
            self._populate_curvature_grad()
            self._pE[np.isnan(self._pE)] = 0
        return self._pE

    @property
    def _mean_edge_length(self):
        return np.mean(self._halfedges['length'][self._halfedges['length'] != -1])

    @property
    def curvature_principal0(self):
        if not np.any(self._k_0):
            self._populate_curvature_grad()
        return self._k_0

    @property
    def curvature_principal1(self):
        if not np.any(self._k_1):
            self._populate_curvature_grad()
        return self._k_1

    @property
    def eigenvector_principal0(self):
        if not np.any(self._e_0):
            self._populate_curvature_grad()
        return self._e_0

    @property
    def eigenvector_principal1(self):
        if not np.any(self._e_1):
            self._populate_curvature_grad()
        return self._e_1

    @property
    def curvature_mean(self):
        if not np.any(self._H):
            self._populate_curvature_grad()
        return self._H
    
    @property
    def curvature_gaussian(self):
        if not np.any(self._K):
            self._populate_curvature_grad()
        return self._K

    def _populate_curvature_grad(self):
        if USE_C:
            self.curvature_grad_c()
        else:
            self.curvature_grad()
        
        if self.smooth_curvature:    
            self._H = self.smooth_per_vertex_data(self._H)
            self._K = self.smooth_per_vertex_data(self._K)
            self._k_0 = self.smooth_per_vertex_data(self._k_0)
            self._k_1 = self.smooth_per_vertex_data(self._k_1)

    def _initialize_curvature_vectors(self):
        sz = self._vertices.shape[0]
        self._H = np.zeros(sz, dtype=np.float32)
        self._K = np.zeros(sz, dtype=np.float32)
        self._E = np.zeros(sz, dtype=np.float32)
        self._k_0 = np.zeros(sz, dtype=np.float32)
        self._k_1 = np.zeros(sz, dtype=np.float32)
        self._e_0 = np.zeros((sz, 3), dtype=np.float32)
        self._e_1 = np.zeros((sz, 3), dtype=np.float32)
        self._pE = np.zeros(sz, dtype=np.float32)
        self._dH = np.zeros(sz, dtype=np.float32)
        self._dK = np.zeros(sz, dtype=np.float32)
        self._dE_neighbors = np.zeros(sz, dtype=np.float32)

        self._set_ck_0(self._k_0)
        self._set_ck_1(self._k_1)
        _flat_e_0 = self._e_0.ravel()
        self._set_ce_0(_flat_e_0)
        _flat_e_1 = self._e_1.ravel()
        self._set_ce_1(_flat_e_1)
        self._set_cH(self._H)
        self._set_cK(self._K)
        self._set_cE(self._E)
        self._set_cpE(self._pE)
        self._set_cdH(self._dH)
        self._set_cdK(self._dK)
        self._set_cdE_neighbors(self._dE_neighbors)

    def _set_ck_0(self, float[:] vec):
        self._ck_0 = &vec[0]

    def _set_ck_1(self, float[:] vec):
        self._ck_1 = &vec[0]

    def _set_ce_0(self, float[:] vec):
        self._ce_0 = &vec[0]

    def _set_ce_1(self, float[:] vec):
        self._ce_1 = &vec[0]

    def _set_cH(self, float[:] vec):
        self._cH = &vec[0]
    
    def _set_cK(self, float[:] vec):
        self._cK = &vec[0]

    def _set_cE(self, float[:] vec):
        self._cE = &vec[0]

    def _set_cpE(self, float[:] vec):
        self._cpE = &vec[0]

    def _set_cdH(self, float[:] vec):
        self._cdH = &vec[0]

    def _set_cdK(self, float[:] vec):
        self._cdK = &vec[0]

    def _set_cdE_neighbors(self, float[:] vec):
        self._cdE_neighbors = &vec[0]

    def remesh(self, n=5, target_edge_length=-1, l=0.5, n_relax=10):
        TriangleMesh.remesh(self, n=n, target_edge_length=target_edge_length, l=l, n_relax=n_relax)
        # Reset H, E, K values
        # self._H = None
        # self._K = None
        # self._E = None
        self._initialize_curvature_vectors()

    cdef int skeleton_edge_split(self, np.int32_t _curr, np.int32_t * new_edges, np.int32_t * new_vertices, np.int32_t * new_faces, int n_edge_idx, int n_vertex_idx, int n_face_idx,  
                            bint live_update=1, bint upsample=0):
        """
        Split triangles evenly along an edge specified by halfedge index _curr.

        Parameters
        ----------
            _curr : int
                Pointer to halfedge defining edge to split.
            live_update : bool
                Update associated faces and vertices after split. Set to False
                to handle this externally (useful if operating on multiple, 
                disjoint edges).
            upsample: bool
                Are we doing loop subdivision? If so, keep track of all edges
                incident on both a new vertex and an old verex that do not
                split an existing edge.
        """
        cdef halfedge_t *curr_edge
        cdef halfedge_t *twin_edge
        cdef np.int32_t _prev, _twin, _next, _twin_prev, _twin_next, _face_1_idx, _face_2_idx, _he_0_idx, _he_1_idx, _he_2_idx, _he_3_idx, _he_4_idx, _he_5_idx, _vertex_idx
        cdef bint interior
        cdef np.int32_t v0, v1
        cdef int i
        cdef np.float32_t x0x, x0y, x0z, x1x, x1y, x1z, n0x, n0y, n0z, n1x, n1y, n1z, ndot
        cdef np.float32_t[VECTORSIZE] _vertex

        if _curr == -1:
            return 0

        if self._chalfedges[_curr].locally_manifold == 0:
            return 0
        
        curr_edge = &self._chalfedges[_curr]
        _prev = curr_edge.prev
        _next = curr_edge.next

        # Grab the new vertex position
        v0 = curr_edge.vertex
        v1 = self._chalfedges[_prev].vertex
        x0x = self._cvertices[v0].position0
        x0y = self._cvertices[v0].position1
        x0z = self._cvertices[v0].position2
        x1x = self._cvertices[v1].position0
        x1y = self._cvertices[v1].position1
        x1z = self._cvertices[v1].position2

        _vertex_idx = new_vertices[n_vertex_idx]

        # FROM HERE

        # displacement along the x1-x0 direction
        #print(curr_edge.length)
        if curr_edge.length > 1e-6:
            # project the _next vertex perpendicularly onto the edge
            n0x = x1x - x0x
            n0y = x1y - x0y
            n0z = x1z - x0z
            n1x = self._cvertices[self._chalfedges[_next].vertex].position0 - x0x
            n1y = self._cvertices[self._chalfedges[_next].vertex].position1 - x0y
            n1z = self._cvertices[self._chalfedges[_next].vertex].position2 - x0z
            ndot = ((n0x*n1x)+(n0y*n1y)+(n0z*n1z))/(curr_edge.length*curr_edge.length)
        else:
            ndot = 0.5

        _vertex[0] = x0x + (x1x-x0x)*ndot
        _vertex[1] = x0y + (x1y-x0y)*ndot
        _vertex[2] = x0z + (x1z-x0z)*ndot

        # TO HERE

        self._cvertices[_vertex_idx].position0 = _vertex[0]
        self._cvertices[_vertex_idx].position1 = _vertex[1]
        self._cvertices[_vertex_idx].position2 = _vertex[2]
        #_vertex_idx = self._new_vertex(_vertex)

        _twin = curr_edge.twin
        interior = (_twin != -1)  # Are we on a boundary?
        
        if interior:
            twin_edge = &self._chalfedges[_twin]
            _twin_prev = twin_edge.prev
            _twin_next = twin_edge.next
        
        # Ensure the original faces have the correct pointers and add two new faces
        self._cfaces[curr_edge.face].halfedge = _curr
        if interior:
            self._cfaces[twin_edge.face].halfedge = _twin
            _face_1_idx = new_faces[n_face_idx] #self._new_face(_twin_prev)
            n_face_idx += 1
            self._cfaces[_face_1_idx].halfedge = _twin_prev
            self._chalfedges[_twin_prev].face = _face_1_idx
        
        _face_2_idx = new_faces[n_face_idx]
        n_face_idx += 1 #self._new_face(_next)
        self._cfaces[_face_2_idx].halfedge = _next
        self._chalfedges[_next].face = _face_2_idx

        # Insert the new faces
        _he_0_idx = new_edges[n_edge_idx]
        n_edge_idx += 1
        _he_4_idx = new_edges[n_edge_idx]
        n_edge_idx += 1
        _he_5_idx = new_edges[n_edge_idx]
        n_edge_idx += 1
        
        if interior:
            _he_1_idx = new_edges[n_edge_idx]
            n_edge_idx += 1
            _he_2_idx = new_edges[n_edge_idx]
            n_edge_idx += 1
            _he_3_idx = new_edges[n_edge_idx]
            n_edge_idx += 1

            self._populate_edge(_he_1_idx, _vertex_idx, prev=_twin_next, next=_twin, face=self._chalfedges[_twin].face, twin=_he_2_idx)
            self._populate_edge(_he_2_idx, self._chalfedges[_twin_next].vertex, prev= _he_3_idx, next=_twin_prev, face=_face_1_idx, twin=_he_1_idx)
            self._populate_edge(_he_3_idx,_vertex_idx, prev=_twin_prev, next=_he_2_idx, face=_face_1_idx, twin=_he_4_idx)
        else:
            _he_1_idx = -1
            _he_2_idx = -1
            _he_3_idx = -1
        
        self._populate_edge(_he_0_idx, self._chalfedges[_next].vertex, prev=_curr, next=_prev, face=self._chalfedges[_curr].face, twin=_he_5_idx)
        self._populate_edge(_he_4_idx, self._chalfedges[_curr].vertex, prev=_he_5_idx, next=_next, face=_face_2_idx, twin=_he_3_idx)
        self._populate_edge(_he_5_idx, _vertex_idx, prev=_next, next=_he_4_idx, face=_face_2_idx, twin=_he_0_idx)

        # Update _prev, next
        self._chalfedges[_prev].prev = _he_0_idx
        self._chalfedges[_next].prev = _he_4_idx
        self._chalfedges[_next].next = _he_5_idx

        if interior:
            # Update _twin_next, _twin_prev
            self._chalfedges[_twin_next].next = _he_1_idx
            self._chalfedges[_twin_prev].prev = _he_2_idx
            self._chalfedges[_twin_prev].next = _he_3_idx

            self._chalfedges[_twin].prev = _he_1_idx
        # Update _curr and _twin
        self._chalfedges[_curr].vertex = _vertex_idx
        self._chalfedges[_curr].next = _he_0_idx

        # Update halfedges
        if interior:
            self._cvertices[self._chalfedges[_he_2_idx].vertex].halfedge = _he_1_idx
        
        self._cvertices[self._chalfedges[_prev].vertex].halfedge = _curr
        self._cvertices[self._chalfedges[_he_4_idx].vertex].halfedge = _next
        self._cvertices[_vertex_idx].halfedge = _he_4_idx
        self._cvertices[self._chalfedges[_he_0_idx].vertex].halfedge = _he_5_idx

        if upsample:
            # Make sure these edges emanate from the new vertex stored at _vertex_idx
            if interior:
                self._loop_subdivision_flip_edges.extend([_he_2_idx])
            
            self._loop_subdivision_flip_edges.extend([_he_0_idx])
            self._loop_subdivision_new_vertices.extend([_vertex_idx])
        
        #print(_he_0_idx, _he_1_idx, _he_2_idx, _he_3_idx, _he_4_idx, _he_5_idx)
        #print(_vertex_idx)
        #print(_face_1_idx, _face_2_idx)

        #print('update')
        if live_update:
            if interior:
                #self._update_face_normals([self._chalfedges[_he_0_idx].face, self._chalfedges[_he_1_idx].face, self._chalfedges[_he_2_idx].face, self._chalfedges[_he_4_idx].face])
                #self._update_vertex_neighbors([self._chalfedges[_curr].vertex, self._chalfedges[_twin].vertex, self._chalfedges[_he_0_idx].vertex, self._chalfedges[_he_2_idx].vertex, self._chalfedges[_he_4_idx].vertex])
            
                update_face_normal(self._chalfedges[_he_0_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_1_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_2_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_4_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                
                #print('vertex_neighbours')
                update_single_vertex_neighbours(self._chalfedges[_curr].vertex, self._chalfedges, self._cvertices, self._cfaces)
                #print('n1')
                update_single_vertex_neighbours(self._chalfedges[_twin].vertex, self._chalfedges, self._cvertices, self._cfaces)
                #print('n2')
                update_single_vertex_neighbours(self._chalfedges[_he_0_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                #print('n3')
                update_single_vertex_neighbours(self._chalfedges[_he_2_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                #print('n')
                update_single_vertex_neighbours(self._chalfedges[_he_4_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                #print('vertex_neighbours done')
            
            else:
                #self._update_face_normals([self._chalfedges[_he_0_idx].face, self._chalfedges[_he_4_idx].face])
                #self._update_vertex_neighbors([self._chalfedges[_curr].vertex, self._chalfedges[_prev].vertex, self._chalfedges[_he_0_idx].vertex, self._chalfedges[_he_4_idx].vertex])
                
                update_face_normal(self._chalfedges[_he_0_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_4_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                
                update_single_vertex_neighbours(self._chalfedges[_curr].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_prev].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_he_0_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_he_4_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
            
            self._clear_flags()
        
        return 1

    cdef int skeleton_split_edges(self, float max_triangle_angle=1.9198622):
        cdef int split_count = 0
        cdef int i
        cdef int n_halfedges = self._halfedges.shape[0]
        cdef float *v0
        cdef float *v1
        cdef float *v2
        cdef float *v3
        cdef float *v4
        cdef float *v5
        cdef float t0d, t1d, t0l, t1l
        cdef float ct0d, ct1d
        cdef int n_edge_idx, n_face_idx, n_vertex_idx

        cdef int* edges_to_split = <int*>PyMem_Malloc(n_halfedges*sizeof(int))
        if not edges_to_split:
            raise MemoryError()
        cdef int* twin_split = <int*>PyMem_Malloc(n_halfedges*sizeof(int))
        if not twin_split:
            raise MemoryError()

        for i in range(n_halfedges):
            twin_split[i] = 0
        
        for i in range(n_halfedges):
            if (not twin_split[i]) and (self._chalfedges[i].vertex != -1):
                v0 = &self._cvertices[self._chalfedges[i].vertex].position0
                v1 = &self._cvertices[self._chalfedges[self._chalfedges[i].next].vertex].position0
                v2 = &self._cvertices[self._chalfedges[self._chalfedges[i].prev].vertex].position0
                t0d = (v0[0]-v1[0])*(v2[0]-v1[0]) + (v0[1]-v1[1])*(v2[1]-v1[1]) + (v0[2]-v1[2])*(v2[2]-v1[2])
                t0l = self._chalfedges[self._chalfedges[i].next].length*self._chalfedges[self._chalfedges[i].prev].length
                
                if t0l > 0:
                    t0d /= t0l
                else:
                    # do not split an edge of length 0
                    continue

                if (self._chalfedges[i].twin != -1):
                    v3 = &self._cvertices[self._chalfedges[self._chalfedges[i].twin].vertex].position0
                    v4 = &self._cvertices[self._chalfedges[self._chalfedges[self._chalfedges[i].twin].next].vertex].position0
                    v5 = &self._cvertices[self._chalfedges[self._chalfedges[self._chalfedges[i].twin].prev].vertex].position0
                
                    t1d = (v3[0]-v4[0])*(v5[0]-v4[0]) + (v3[1]-v4[1])*(v5[1]-v4[1]) + (v3[2]-v4[2])*(v5[2]-v4[2])
                    t1l = self._chalfedges[self._chalfedges[self._chalfedges[i].twin].next].length*self._chalfedges[self._chalfedges[self._chalfedges[i].twin].prev].length

                    if t1l > 0:
                        t1d /= t1l
                    else:
                        # do not split an edge of length 0
                        continue
                else:
                    t1d = 0

                t0d = np.clip(t0d, -1, 1)
                t1d = np.clip(t1d, -1, 1)

                ct0d = np.arccos(t0d)
                ct1d = np.arccos(t1d)

                if (ct1d < max_triangle_angle) or (ct0d < max_triangle_angle):
                    continue

                # split according to the larger of the two triangles
                # (skeleton_edge_split relies on projection from one of the triangles)
                if (ct0d > ct1d):
                    #print(f"Splitting {i}")
                    if self._chalfedges[i].twin != -1:
                        twin_split[self._chalfedges[i].twin] = 1
                    edges_to_split[split_count] = i
                else:
                    #print(f"Splitting {self._chalfedges[i].twin}")
                    twin_split[i] = 1
                    edges_to_split[split_count] = self._chalfedges[i].twin
                split_count += 1
        
        n_edges = self.new_edges(int(split_count*6))
        n_edge_idx = 0
        n_faces = self.new_faces(int(split_count*2))
        n_face_idx = 0
        n_vertices = self.new_vertices(int(split_count))
        n_vertex_idx = 0
        #print(self._halfedges[n_edges])

        for i in range(split_count):
            e = edges_to_split[i]
            #print(i, e, n_edge_idx, n_edges)
            self.skeleton_edge_split(e, 
                                <np.int32_t *> np.PyArray_DATA(n_edges), 
                                <np.int32_t *> np.PyArray_DATA(n_vertices), 
                                <np.int32_t *> np.PyArray_DATA(n_faces), 
                                n_edge_idx, n_vertex_idx, n_face_idx)
            #self.edge_split(e)
            n_edge_idx += 6
            n_face_idx += 2
            n_vertex_idx += 1

        PyMem_Free(edges_to_split)
        PyMem_Free(twin_split)
                
        print('Split count: %d' % (split_count))
        return split_count

    cpdef int skeleton_edge_collapse(self, np.int32_t _curr, bint live_update=1):
        """
        A.k.a. delete two triangles. Remove an edge, defined by halfedge _curr,
        and its associated triangles, while keeping mesh connectivity.

        TODO: A known problem is, in the presence of boundaries, edge_collapse()
        will eventually (after 3-5 remesh steps) produce a few singular edges.

        Parameters
        ----------
            _curr: int
                Pointer to halfedge defining edge to collapse.
            live_update : bool
                Update associated faces and vertices after collapse. Set to 
                False to handle this externally (useful if operating on 
                multiple, disjoint edges).
        """

        cdef halfedge_t *curr_halfedge
        cdef halfedge_t *twin_halfedge
        cdef np.int32_t _prev, _twin, _next, _prev_twin, _prev_twin_vertex, _next_prev_twin, _next_prev_twin_vertex, _twin_next_vertex, _next_twin_twin_next, _next_twin_twin_next_vertex, vl, vd, _dead_vertex, _live_vertex, vn, vtn, face0, face1, face2, face3
        cdef np.int32_t *neighbours_live
        cdef np.int32_t *neighbours_dead
        cdef np.int32_t shared_vertex
        cdef np.float32_t px, py, pz
        cdef bint fast_collapse_bool, interior
        cdef int i, j, twin_count, dead_count
        cdef np.int32_t[5*NEIGHBORSIZE] dead_vertices

        if _curr == -1:
            return 0

        if self._chalfedges[_curr].locally_manifold == 0:
            return 0

        # Create the pointers we need
        curr_halfedge = &self._chalfedges[_curr]
        _prev = curr_halfedge.prev
        _next = curr_halfedge.next
        _twin = curr_halfedge.twin

        if (self._chalfedges[_next].twin == -1) or (self._chalfedges[_prev].twin == -1):
            # Collapsing this edge will create another free edge
            return 0

        interior = (_twin != -1)  # Are we on a boundary?

        if interior:
            #nn = self._vertices['neighbors'][curr_halfedge.vertex]
            #nn_mask = (nn != -1)
            #if (-1 in self._halfedges['twin'][nn]*nn_mask):
            #    return
            
            if not self._check_neighbour_twins(curr_halfedge.vertex):
                return 0

            #nn = self._vertices['neighbors'][self._halfedges['vertex'][_twin]]
            #nn_mask = (nn != -1)
            #if (-1 in self._halfedges['twin'][nn]*nn_mask):
            #    return
            
            if not self._check_neighbour_twins(self._chalfedges[_twin].vertex):
                return 0

            twin_halfedge = &self._chalfedges[_twin]
            _twin_prev = twin_halfedge.prev
            _twin_next = twin_halfedge.next

            if (self._chalfedges[_twin_prev].twin == -1) or (self._chalfedges[_twin_next].twin == -1):
                # Collapsing this edge will create another free edge
                return 0

        _dead_vertex = self._chalfedges[_prev].vertex
        _live_vertex = curr_halfedge.vertex

        # Grab the valences of the 4 points near the edge
        if interior:
            vn, vtn = self._cvertices[self._chalfedges[_next].vertex].valence, self._cvertices[self._chalfedges[_twin_next].vertex].valence

            # Make sure we create no vertices of valence <3 (manifoldness)
            # ((vl + vd - 3) < 4) or 
            #if (vn < 4) or (vtn < 4):
            #    return 0

        vl, vd = self._cvertices[_live_vertex].valence, self._cvertices[_dead_vertex].valence
        
        #if ((vl + vd - 4) < 4):
        #    return 0
        
        cdef bint locally_manifold = self._cvertices[_live_vertex].locally_manifold and self._cvertices[_dead_vertex].locally_manifold

        # Check for creation of multivalent edges and prevent this (manifoldness)
        fast_collapse_bool = (locally_manifold and (vl < NEIGHBORSIZE) and (vd < NEIGHBORSIZE))
        if fast_collapse_bool:
            # Do it the fast way if we can
            neighbours_live = &self._cvertices[_live_vertex].neighbor0
            neighbours_dead = &self._cvertices[_dead_vertex].neighbor0
            twin_count = 0
            shared_vertex = -1
            for i in range(NEIGHBORSIZE):
                if neighbours_live[i] == -1:
                    break
                for j in range(NEIGHBORSIZE):
                    if neighbours_dead[j] == -1:
                        break
                    if self._chalfedges[neighbours_live[i]].vertex == self._chalfedges[neighbours_dead[j]].vertex:
                        if twin_count > 2:
                            break
                        if (twin_count == 0) or ((twin_count > 0) and (self._chalfedges[neighbours_dead[j]].vertex != shared_vertex)):
                            shared_vertex = self._chalfedges[neighbours_live[i]].vertex
                            twin_count += 1
                if twin_count > 2:
                    break

            # no more than two vertices shared by the neighbors of dead and live vertex
            if twin_count != 2:
                return 0

            # assign
            for i in range(NEIGHBORSIZE):
                if (neighbours_dead[i] == -1):
                    continue
                self._chalfedges[self._chalfedges[neighbours_dead[i]].twin].vertex = _live_vertex
                self._chalfedges[self._chalfedges[neighbours_dead[i]].prev].vertex = _live_vertex

            #live_nn = self._vertices['neighbors'][_live_vertex]
            #dead_nn =  self._vertices['neighbors'][_dead_vertex]
            #live_mask = (live_nn != -1)
            #dead_mask = (dead_nn != -1)
            #live_list = self._halfedges['vertex'][live_nn[live_mask]]
            #dead_list = self._halfedges['vertex'][dead_nn[dead_mask]]
        else:
            # grab the set of halfedges pointing to dead_vertices
            
            dead_count = 0
            for i in range(self._halfedges.shape[0]):
                if self._chalfedges[i].vertex == _dead_vertex:
                    dead_vertices[dead_count] = i
                    dead_count += 1
                    if dead_count > 5*NEIGHBORSIZE:
                        print(f'WARNING: Way too many dead vertices: {dead_count}! Politely declining to collapse.')
                        return 0

            # loop over all live vertices and check for twins in dead_vertices,
            # as we do in fast_collapse
            twin_count = 0
            shared_vertex = -1
            for i in range(self._halfedges.shape[0]):
                if self._chalfedges[i].twin == -1:
                    continue
                if self._chalfedges[i].vertex == _live_vertex:
                    for j in range(dead_count):
                        if self._chalfedges[dead_vertices[j]].twin == -1:
                            continue
                        if self._chalfedges[self._chalfedges[i].twin].vertex == self._chalfedges[self._chalfedges[dead_vertices[j]].twin].vertex:
                            if twin_count > 2:
                                break
                            if (twin_count == 0) or ((twin_count > 0) and (self._chalfedges[self._chalfedges[dead_vertices[j]].twin].vertex != shared_vertex)):
                                shared_vertex = self._chalfedges[self._chalfedges[i].twin].vertex
                                twin_count += 1
                    if twin_count > 2:
                        break

            # no more than two vertices shared by the neighbors of dead and live vertex
            if twin_count != 2:
                return 0

            # assign
            for i in range(dead_count):
                self._chalfedges[dead_vertices[i]].vertex = _live_vertex

            #twin_mask = (self._halfedges['twin'] != -1)
            #dead_mask = (self._halfedges['vertex'] == _dead_vertex)
            #dead_list = self._halfedges['vertex'][self._halfedges['twin'][dead_mask & twin_mask]]
            #live_list = self._halfedges['vertex'][self._halfedges['twin'][(self._halfedges['vertex'] == _live_vertex) & twin_mask]]

        # twin_list = list((set(dead_list) & set(live_list)) - set([-1]))
        # if len(twin_list) != 2:
        #     return 0
            
        # Collapse to the midpoint of the original edge vertices
        # if fast_collapse_bool:
        #    self._halfedges['vertex'][self._halfedges['twin'][dead_nn[dead_mask]]] = _live_vertex
        # else:
        #     self._halfedges['vertex'][dead_mask] = _live_vertex
        
        # if fast_collapse_bool:
        #     if not self._check_collapse_fast(_live_vertex, _dead_vertex):
        #         return
        # else:
        #     if not self._check_collapse_slow(_live_vertex, _dead_vertex):
        #         return
        
        # _live_pos = self._vertices['position'][_live_vertex]
        # _dead_pos = self._vertices['position'][_dead_vertex]
        # self._vertices['position'][_live_vertex] = 0.5*(_live_pos + _dead_pos)

        #px = 0.5*(self._cvertices[_live_vertex].position0 + self._cvertices[_dead_vertex].position0)
        #py = 0.5*(self._cvertices[_live_vertex].position1 + self._cvertices[_dead_vertex].position1)
        #pz = 0.5*(self._cvertices[_live_vertex].position2 + self._cvertices[_dead_vertex].position2)
        self._cvertices[_live_vertex].position0 = self._cvertices[_live_vertex].position0
        self._cvertices[_live_vertex].position1 = self._cvertices[_live_vertex].position1
        self._cvertices[_live_vertex].position2 = self._cvertices[_live_vertex].position2
        
        # update valence of vertex we keep
        self._cvertices[_live_vertex].valence = vl + vd - 3
        
        # delete dead vertex
        self._vertices[_dead_vertex] = -1
        self._vertex_vacancies.append(_dead_vertex)

        # Zipper the remaining triangles
        self._zipper(_next, _prev)
        if interior:
            self._zipper(_twin_next, _twin_prev)
        # We need some more pointers
        # TODO: make these safer
        _prev_twin = self._chalfedges[_prev].twin
        _prev_twin_vertex = self._chalfedges[_prev_twin].vertex
        _next_prev_twin = self._chalfedges[_prev_twin].next
        _next_prev_twin_vertex = self._chalfedges[_next_prev_twin].vertex
        if interior:
            _twin_next_vertex = self._chalfedges[_twin_next].vertex
            _next_twin_twin_next = self._chalfedges[self._chalfedges[_twin_next].twin].next
            _next_twin_twin_next_vertex = self._chalfedges[_next_twin_twin_next].vertex
            
        # Make sure we have good _vertex_halfedges references
        self._cvertices[_live_vertex].halfedge = _prev_twin
        self._cvertices[_prev_twin_vertex].halfedge = _next_prev_twin
        if interior:
            self._cvertices[_twin_next_vertex].halfedge = self._chalfedges[_twin_next].twin
            self._cvertices[_next_twin_twin_next_vertex].halfedge = self._chalfedges[_next_twin_twin_next].next

        # Grab faces to update
        face0 = self._chalfedges[self._chalfedges[_next].twin].face
        face1 = self._chalfedges[self._chalfedges[_prev].twin].face
        if interior:
            face2 = self._chalfedges[self._chalfedges[_twin_next].twin].face
            face3 = self._chalfedges[self._chalfedges[_twin_prev].twin].face

        # Delete the inner triangles
        self._face_delete(_curr)
        if interior:
            self._face_delete(_twin)

        try:
            if live_update:
                if interior:
                    # Update faces
                    #self._update_face_normals([face0, face1, face2, face3])
                    #self._update_vertex_neighbors([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])
                    
                    update_face_normal(face0, self._chalfedges, self._cvertices, self._cfaces)
                    update_face_normal(face1, self._chalfedges, self._cvertices, self._cfaces)
                    update_face_normal(face2, self._chalfedges, self._cvertices, self._cfaces)
                    update_face_normal(face3, self._chalfedges, self._cvertices, self._cfaces)
                    
                    update_single_vertex_neighbours(_live_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    update_single_vertex_neighbours(_prev_twin_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    update_single_vertex_neighbours(_next_prev_twin_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    update_single_vertex_neighbours(_twin_next_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    
                else:
                    #self._update_face_normals([face0, face1])
                    #self._update_vertex_neighbors([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex])
                    
                    update_face_normal(face0, self._chalfedges, self._cvertices, self._cfaces)
                    update_face_normal(face1, self._chalfedges, self._cvertices, self._cfaces)
                    
                    update_single_vertex_neighbours(_live_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    update_single_vertex_neighbours(_prev_twin_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    update_single_vertex_neighbours(_next_prev_twin_vertex, self._chalfedges, self._cvertices, self._cfaces)
                    
                self._clear_flags()
        
        except RuntimeError as e:
            print(_curr, _twin, _next, _prev, _twin_next, _twin_prev, _next_prev_twin, _next_twin_twin_next, _prev_twin)
            print([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])
            raise e
        
        return 1

    def skeleton_collapse_edges(self, float collapse_threshold):
        cdef int collapse_count = 0
        cdef int collapse_fails = 0
        cdef int i
        cdef int n_halfedges = self._halfedges.shape[0]
        cdef float *n1
        cdef float *n2
        cdef float nd
        
        #find which vertices are locally manifold
        self._update_vertex_locally_manifold()
        
        for i in range(n_halfedges):
            if (self._chalfedges[i].vertex != -1) and (self._chalfedges[i].length < collapse_threshold):
                collapse_ret = self.skeleton_edge_collapse(i)
                collapse_count += collapse_ret
                collapse_fails += (1-collapse_ret)
        print('Collapse count: ' + str(collapse_count) + '[' + str(collapse_fails) +' failed]')
        
        return collapse_count

    cdef int skeleton_remesh(self, float target_edge_length=-1, float max_triangle_angle=1.9198622):
        cdef int k, i, ct
        cdef float collapse_threshold, xl, xu, yl, yu, zl, zu, diag  #, split_threshold

        cdef int n_halfedges = self._halfedges.shape[0]
        cdef int n_vertices = self._vertices.shape[0]

        if (target_edge_length < 0):
            # Guess edge_length
            xl, yl, zl, xu, yu, zu = self.bbox
            diag = math.sqrt((xu-xl)*(xu-xl)+(yu-yl)*(yu-yl)+(zu-zl)*(zu-zl))
            collapse_threshold = 0.002*diag
        else:
            collapse_threshold = target_edge_length

        #split_threshold = 1.66*collapse_threshold

        self._update_singular_edge_locally_manifold()
        self._update_singular_vertex_locally_manifold()
        
        print(f"Target edge length: {collapse_threshold}")
        ct = self.skeleton_collapse_edges(collapse_threshold)

        self._update_singular_edge_locally_manifold()
        self._update_singular_vertex_locally_manifold()

        ct = self.skeleton_split_edges(max_triangle_angle=max_triangle_angle)
        # ct = self.split_edges(split_threshold)

        # Let's double-check the mesh manifoldness
        self._manifold = None
        self.manifold
        self._initialize_curvature_vectors()

        return 1

    def _compute_curvature_tensor_eig(self, Mvi):
        """
        Return the first two eigenvalues and eigenvectors of 3x3 curvature 
        tensor. The third eigenvector is the unit normal of the point for
        which the curvature tensor is defined.

        This is a closed-form solution, and it assumes no eigenvalue is 0.

        Parameters
        ----------
            Mvi : np.array
                3x3 curvature tensor at a point.

        Returns
        -------
            l1, l2 : float
                Eigenvalues
            v1, v2 : np.array
                Eigenvectors
        """
        # Solve the eigenproblem in closed form
        m00 = Mvi[0,0]
        m01 = Mvi[0,1]
        m02 = Mvi[0,2]
        m11 = Mvi[1,1]
        m12 = Mvi[1,2]
        m22 = Mvi[2,2]

        # Here we use the fact that Mvi is symnmetric and we know
        # one of the eigenvalues must be 0
        p = -m00*m11 - m00*m22 + m01*m01 + m02*m02 - m11*m22 + m12*m12
        q = m00 + m11 + m22
        r = np.sqrt(4*p + q*q)
        
        # Eigenvalues
        l1 = 0.5*(q-r)
        l2 = 0.5*(q+r)

        def safe_divide(x, y):
            # if y == 0:
            #     return 0
            return (y!=0)*1.*x/y

        # Now calculate the eigenvectors, assuming x = 1
        z1n = ((m00 - l1)*(m11 - l1) - (m01*m01))
        z1d = (m01*m12 - m02*(m11 - l1))
        z1 = safe_divide(z1n, z1d)
        y1n = (m12*z1 + m01)
        y1d = (m11 - l1)
        y1 = safe_divide(y1n, y1d)
        
        v1 = np.array([1., y1, z1])
        v1_norm = np.sqrt((v1*v1).sum())
        v1 = v1/v1_norm
        
        z2n = ((m00 - l2)*(m11 - l2) - (m01*m01))
        z2d = (m01*m12 - m02*(m11 - l2))
        z2 = safe_divide(z2n, z2d)
        y2n = (m12*z2 + m01)
        y2d = (m11 - l2)
        y2 = safe_divide(y2n, y2d)
        
        v2 = np.array([1., y2, z2])
        v2_norm = np.sqrt((v2*v2).sum())
        v2 = v2/v2_norm

        return l1, l2, v1, v2

    cdef curvature_grad_c(self, float dN=0.1, float skip_prob=0.0):
        dEdN = np.ascontiguousarray(np.zeros((self._vertices.shape[0], 3), dtype=np.float32), dtype=np.float32)
        cdef points_t[:] cdEdN = dEdN.ravel().view(POINTS_DTYPE)
        c_curvature_grad(&(self._cvertices[0]), 
                        &(self._cfaces[0]),
                        &(self._chalfedges[0]),
                        dN,
                        skip_prob,
                        self._vertices.shape[0],
                        &(self._ck_0[0]),
                        &(self._ck_1[0]),
                        &(self._ce_0[0]),
                        &(self._ce_1[0]),
                        &(self._cH[0]),
                        &(self._cK[0]),
                        &(self._cdH[0]),
                        &(self._cdK[0]),
                        &(self._cE[0]),
                        &(self._cpE[0]),
                        &(self._cdE_neighbors[0]),
                        self.kc,
                        self.kg,
                        self.c0,
                        &(cdEdN[0]))
        return dEdN

    cdef curvature_grad(self, float dN=0.1, float skip_prob=0.0):
        """
        Estimate curvature. Here we follow a mix of ESTIMATwG THE 
        TENSOR OF CURVATURE OF A SURFACE FROM A POLYHEDRAL 
        APPROXIMATION by Gabriel Taubin from Proceedings of IEEE 
        International Conference on Computer Vision, June 1995 and 
        Estimating the Principal Curvatures and the Darboux Frame 
        From Real 3-D Range Data by Eyal Hameiri and Ilan Shimshon 
        from IEEE TRANSACTIONS ON SYSTEMS, MAN, AND CYBERNETICS-PART
        B: CYBERNETICS, VOL. 33, NO. 4, AUGUST 2003

        Units of energy are in kg*nm^2/s^2
        """

        cdef int iv
        cdef float l1, l2
        # cdef float[3] v1
        # cdef float[3] v2
        v1 = np.zeros(3, dtype=np.float32)
        v2 = np.zeros(3, dtype=np.float32)
        cdef float[:] v1_view = v1
        cdef float[:] v2_view = v2
        
        Mvi = np.zeros((3,3), dtype=np.float32)
        cdef float[:,:] Mvi_view = Mvi

        # H = np.zeros(self._vertices.shape[0])
        # K = np.zeros(self._vertices.shape[0])
        # dH = np.zeros(self._vertices.shape[0])
        # dK = np.zeros(self._vertices.shape[0])
        # dE_neighbors = np.zeros(self._vertices.shape[0])
        areas = np.zeros(self._vertices.shape[0], dtype=np.float32)
        # skip = np.random.rand(self._vertices.shape[0])
        for iv in range(self._vertices.shape[0]):
            if self._cvertices[iv].halfedge == -1:
                self._k_0[iv] = 0.0
                self._k_1[iv] = 0.0
                self._H[iv] = 0.0
                self._K[iv] = 0.0
                self._dH[iv] = 0.0
                self._dK[iv] = 0.0
                self._dE_neighbors[iv] = 0.0
                continue
            # Monte carlo selection of vertices to update
            # Stochastically choose which vertices to adjust
            if (skip_prob > 0) and (np.random.rand() < skip_prob):
                self._k_0[iv] = 0.0
                self._k_1[iv] = 0.0
                self._H[iv] = 0.0
                self._K[iv] = 0.0
                self._dH[iv] = 0.0
                self._dK[iv] = 0.0
                self._dE_neighbors[iv] = 0.0
                continue
            # Vertex and its normal
            vi = self._vertices['position'][iv,:]  # nm
            Nvi = self._vertices['normal'][iv,:]  # unitless

            p = I - Nvi[:,None]*Nvi[None,:]  # unitless
                
            # vertex nearest neighbors
            neighbors = self._vertices['neighbors'][iv]
            neighbor_mask = (neighbors != -1)
            neighbor_vertices = self._halfedges['vertex'][neighbors]
            vjs = self._vertices['position'][neighbor_vertices[neighbor_mask]]  # nm
            Nvjs = self._vertices['normal'][neighbor_vertices[neighbor_mask]]  # unitless

            # Neighbor vectors & displaced neighbor tangents
            dvs = vjs - vi[None,:]  # nm
            dvs_1 = dvs - (Nvi*dN)[None, :]  # nm

            # radial weighting
            r_sum = np.sum(1./np.sqrt((dvs*dvs).sum(1)))  # 1/nm

            # Norms
            dvs_norm = np.sqrt((dvs*dvs).sum(1))  # nm
            dvs_1_norm = np.sqrt((dvs_1*dvs_1).sum(1))  # nm

            # Hats
            dvs_hat = dvs/dvs_norm[:,None]  # unitless
            dvs_1_hat = dvs_1/dvs_1_norm[:,None]  # unitless

            # Tangents
            T_thetas = np.dot(p,-dvs.T).T  # nm^2
            Tijs = T_thetas/np.sqrt((T_thetas*T_thetas).sum(1)[:,None])  # nm
            Tijs[np.sum(T_thetas,axis=1) == 0, :] = 0

            # Edge normals subtracted from vertex normals
            # FIXME: If Nvi = [-1,-1,-1] and the surrounding area is flat the inner square  
            #        root will produce a nan. This only happens on non-manifold meshes.
            Ni_diffs = np.sqrt(2. - 2.*np.sqrt(1.-((Nvi[None,:]*dvs_hat).sum(1))**2))  # unitless 
            Nj_diffs = np.sqrt(2. - 2.*np.sqrt(1.-((Nvjs*dvs_hat).sum(1))**2))  # unitless
            Nj_1_diffs = np.sqrt(2. - 2.*np.sqrt(1.-((Nvjs*dvs_1_hat).sum(1))**2))  # unitless

            # Compute the principal curvatures from the difference in normals (same as difference in tangents)
            kjs = 2.*Nj_diffs/dvs_norm  # 1/nm
            kjs_1 = 2.*Nj_1_diffs/dvs_1_norm  # 1/nm

            k = 2.*np.sign((Nvi[None,:]*(-dvs)).sum(1))*Ni_diffs/dvs_norm  # 1/nm
            w = (1./dvs_norm)/r_sum  # unitless

            # Calculate areas
            Aj = self._faces['area'][self._halfedges['face'][neighbors[neighbor_mask]]]  # nm^2
            areas[iv] = np.sum(Aj)  # nm^2

            # Compute the change in bending energy along the edge (assumes no perpendicular contributions and thus no Gaussian curvature)
            dEj = Aj*w*self.kc*(2.0*kjs - self.c0)*(kjs_1 - kjs)/dN  # eV/nm
            Mvi[:] = ((w[None,:,None]*k[None,:,None]*Tijs.T[:,:,None]*Tijs[None,:,:]).sum(axis=1)).astype(np.float32)  # nm

            # l1, l2, v1, v2 = self._compute_curvature_tensor_eig(Mvi)
            fcompute_curvature_tensor_eig(&Mvi_view[0,0], &l1, &l2, &v1_view[0], &v2_view[0])

            # Eigenvectors
            m = np.vstack([v1, v2, Nvi]).T  # nm, nm, unitless

            # Principal curvatures
            self._k_0[iv] = 3.*l1 - l2  # 1/nm
            self._k_1[iv] = 3.*l2 - l1  # 1/nm

            # Mean and Gaussian curvatures
            self._H[iv] = 0.5*(self._k_0[iv] + self._k_1[iv])  # 1/nm
            self._K[iv] = self._k_0[iv]*self._k_1[iv]  # 1/nm^2

            # Now calculate the shift
            # We construct a quadratic in the space of T_1 vs. T_2
            t_1, t_2, _ = np.dot(vjs-vi,m).T  # nm^2
            A = np.array([t_1**2, t_2**2]).T  # nm^2
            
            # Update the equation y-intercept to displace athe curve along
            # the normal direction
            b = np.dot(A,np.array([self._k_0[iv],self._k_1[iv]])) - dN  # nm
            
            # Solve
            # Previously k_p, _, _, _ = np.linalg.lstsq(A, b)
            k_p = np.dot(np.dot(np.linalg.pinv(np.dot(A.T,A)),A.T),b)  # 1/nm

            # Finite differences of displaced curve and original curve
            self._dH[iv] = (0.5*(k_p[0] + k_p[1]) - self._H[iv])/dN  # 1/nm^2
            self._dK[iv] = ((k_p[0]-self._k_0[iv])*self._k_1[iv] + self._k_0[iv]*(k_p[1]-self._k_1[iv]))/dN  # 1/nm

            self._dE_neighbors[iv] = np.sum(dEj)  # eV/nm

        # Calculate Canham-Helfrich energy functional
        self._E = areas*(0.5*self.kc*(2.0*self._H - self.c0)**2 + self.kg*self._K)  # eV

        #self._H = H  # 1/nm
        #self._E = E  # eV
        #self._K = K  # 1/nm^2
        
        self._pE = np.exp(-(1.0/KBT)*self._E)  # unitless
        
        ## Take into account the change in neighboring energies for each
        # vertex shift
        # Compute dEdN by component
        dEdN_H = areas*self.kc*(2.0*self._H-self.c0)*self._dH  # eV/nm
        dEdN_K = areas*self.kg*self._dK  # eV/nm
        dEdN_sum = (dEdN_H + dEdN_K + self._dE_neighbors) # eV/nm # + dE_neighbors)
        dEdN = -1.0*dEdN_sum # eV/nm # *(1.0-self._pE)

        # print('Contributions: {}, {}, {}'.format(np.mean(dEdN_H), np.mean(dEdN_K), np.mean(dE_neighbors)))
        # print('Total energy difference: {} {} {} {}'.format(np.min(dEdN_sum), np.mean(dEdN_sum), np.max(dEdN_sum), np.max(dEdN_sum)-np.min(dEdN_sum)))
        # dEdN = -(4.*self.kc*H*dH + self.kg*dK)*pE
        # dpdN = -250.*np.exp(-250.*E)*dEdN
        
        # Return energy shift along direction of the normal
        return dEdN[:,None]*self._vertices['normal']  # eV/nm

    def point_attraction_grad(self, points, sigma, w=0.95):
        """
        Attractive force of membrane to points.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit.
            sigma : float
                Localization uncertainty of points.
        """
        dirs = []

        # pt_cnt_dist_2 will eventually be a MxN (# points x # vertices) matrix, but becomes so in
        # first loop iteration when we add a matrix to this scalar
        # pt_cnt_dist_2 = 0

        # for j in range(points.shape[1]):
        #     pt_cnt_dist_2 = pt_cnt_dist_2 + (points[:,j][:,None] - self._vertices['position'][:,j][None,:])**2

        charge_sigma = self._mean_edge_length/2.5
        # pt_weight_matrix = 1. - w*np.exp(-pt_cnt_dist_2/(2*charge_sigma**2))
        pt_weight_matrix = np.zeros((points.shape[0], self._vertices.shape[0]), 'f4')
        membrane_mesh_utils.calculate_pt_cnt_dist_2(points, self._vertices, pt_weight_matrix, w, charge_sigma)
        pt_weights = np.prod(pt_weight_matrix, axis=1)
        for i in range(self._vertices.shape[0]): 
            if self._vertices['halfedge'][i] != -1:
                d = self._vertices['position'][i, :] - points
                dd = (d*d).sum(1)
                
                r = np.sqrt(dd)/sigma
                
                rf = -(1-r**2)*np.exp(-r**2/2) + (1-np.exp(-(r-1)**2/2))*(r/(r**3 + 1))

                # Points at the vertex we're interested in are not de-weighted by the
                # pt_weight_matrix
                rf = rf*(pt_weights/pt_weight_matrix[:, i])
                
                attraction = (-d*(rf/np.sqrt(dd))[:,None]).sum(0)
            else:
                attraction = np.array([0,0,0])
            
            dirs.append(attraction)

        dirs = np.vstack(dirs)
        dirs[self._vertices['halfedge'] == -1] = 0

        return dirs

    cdef point_attraction_grad_kdtree(self, points, sigma, float w=0.95, int search_k=200):
        """
        Attractive force of membrane to points.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit (nm).
            sigma : np.array
                Localization uncertainty of points (nm).
            w : float
                Weight (unitless)
            search_k : int
                Number of vertex point neighbors to consider
        """
        cdef int i, n_verts
        # cdef float charge_sigma, charge_var, attraction_norm

        n_verts = self._vertices.shape[0]
        #dirs = []
        dirs = np.zeros((n_verts,3), dtype=np.float32)
        attraction = np.zeros(3, dtype=np.float32)

        # pt_cnt_dist_2 will eventually be a MxN (# points x # vertices) matrix, but becomes so in
        # first loop iteration when we add a matrix to this scalar
        # pt_cnt_dist_2 = 0

        # for j in range(points.shape[1]):
        #     pt_cnt_dist_2 = pt_cnt_dist_2 + (points[:,j][:,None] - self._vertices['position'][:,j][None,:])**2

        charge_sigma = self._mean_edge_length/2.5  # nm
        charge_var = (2*charge_sigma**2)  # nm^2

        # pt_weight_matrix = 1. - w*np.exp(-pt_cnt_dist_2/(2*charge_sigma**2))
        # pt_weights = np.prod(pt_weight_matrix, axis=1)

        if self._tree is None:
            # Compute a KDTree on points
            self._tree = scipy.spatial.cKDTree(points)

        for i in np.arange(n_verts):
            if self._cvertices[i].halfedge == -1:
                continue
                
            dists, neighbors = self._tree.query(self._vertices['position'][i,:], search_k)
            
            #the query might run out of points to query, throw out invalid ones.
            # TODO - restrict search_k to max number of points instead?
            valid = neighbors < self._tree.n
            dists = dists[valid]
            neighbours = neighbors[valid]
            # neighbors = tree.query_ball_point(self._vertices['position'][i,:], search_r)
            
            try:
                d = self._vertices['position'][i,:] - points[neighbors]  # nm
            except(IndexError):
                raise IndexError('Could not access neighbors for position {}.'.format(self._vertices['position'][i,:]))
            dd = (d*d).sum(1)  # nm^2
            # dd = dists*dists

            # if self._puncture_test:
            #     dvn = (self._halfedges['length'][(self._vertices['neighbors'][i])[self._vertices['neighbors'][i] != -1]])**2
            #     if np.min(dvn) < np.min(dd):
            #         self._puncture_candidates.append(i)
            pt_weight_matrix = 1. - w*np.exp(-dd/charge_var)  # unitless
            pt_weights = np.prod(pt_weight_matrix)  # unitless
            r = np.sqrt(dd)/sigma[neighbors]  # unitless
            # r = dists/sigma[neighbors]
            
            rf = -(1-r**2)*np.exp(-r**2/2) + (1-np.exp(-(r-1)**2/2))*(r/(r**3 + 1))  # unitless
            # Points at the vertex we're interested in are not de-weighted by the
            # pt_weight_matrix
            rf = rf*(pt_weights/pt_weight_matrix) # unitless
            
            attraction[:] = (-d*(rf/np.sqrt(dd))[:,None]).sum(0)  # unitless
            # attraction[:] = (-d*(rf/dists)[:,None]).sum(0)  # unitless
            attraction_norm = np.linalg.norm(attraction)
            # attraction_norm = math.sqrt(attraction[0]*attraction[0]+attraction[1]*attraction[1]+attraction[2]*attraction[2])
            attraction[:] = (attraction*np.prod(1-np.exp(-r**2/2)))/attraction_norm  # unitless
            attraction[attraction_norm == 0] = 0  # div by zero
            dirs[i,:] = attraction
            # else:
            #     attraction = np.array([0,0,0])
            
            # dirs.append(attraction)


        # dirs = np.vstack(dirs)
        # dirs[self._vertices['halfedge'] == -1] = 0

        return dirs

    def delaunay_remesh(self, points, eps=1):
        print('Delaunay remesh...')

        # Generate tesselation from mesh control points
        v = self._vertices['position'][self._vertices['halfedge']!=-1]
        d = scipy.spatial.Delaunay(v)
        
        # circumradius = v[d].mean(1)

        # Ensure all simplex vertices are wound s.t. normals point away from simplex centroid
        tri = delaunay_utils.orient_simps(d, v)

        # Remove simplices outside of our mesh
        # ext_inds = delaunay_utils.ext_simps(tri, self)
        ext_inds = delaunay_utils.greedy_ext_simps(tri, self)
        simps = delaunay_utils.del_simps(tri, ext_inds)

        # Remove simplices that do not contain points
        # eps = self._mean_edge_length/5.0  # How far outside of a tetrahedron do we 
                                          # consider a point 'inside' a tetrahedron?
                                          # TODO: /5.0 is empirical. sqrt(6)/4*base length is circumradius
                                          # TODO: account for sigma?
        #print('Guessed eps: {}'.format(eps))
        # empty_inds = delaunay_utils.empty_simps(simps, v, points, eps=eps)
        #empty_inds = delaunay_utils.greedy_empty_simps(simps, self, points, eps=eps)
        #simps_ = delaunay_utils.del_simps(simps, empty_inds)

        # Recover new triangulation
        faces = delaunay_utils.surf_from_delaunay(simps)

        # Make sure we pass in only the vertices used
        old_v, idxs = np.unique(faces.ravel(), return_inverse=True)
        new_v = np.arange(old_v.shape[0])
        reindexed_faces = new_v[idxs].reshape(faces.shape)

        # Rebuild mesh
        self.build_from_verts_faces(v[old_v,:], reindexed_faces, clear=True)

        # Delaunay remeshing has a penchant for flanges
        # self._remove_singularities()
        # self.repair()

        self._initialize_curvature_vectors()

    def _punch_hole(self, np.int32_t face0, np.int32_t face1):
        """
        Create a hole in the mesh by connecting face0 and face1
        with 6 additional triangles to form a triangular prism,
        and then deleting face0 and face1.

        Parameters
        ----------
        face0 : np.int32_t
            Index of face
        face1 : np.int32_t
            Index of opposing face
        """
        cdef int n_edge_idx, n_face_idx

        # Allocate 6 new faces, 18 new edges, no new vertices
        n_faces = self.new_faces(6)
        n_face_idx = 0
        n_edges = self.new_edges(18)
        n_edge_idx = 0

        # Construct boxes in between each pair of edges
        # Each new face has one edge paired with a twin of face0 or face1, no repeats
        self._insert_square(self._cfaces[face0].halfedge, 
                            self._cfaces[face1].halfedge,  
                            <np.int32_t *> np.PyArray_DATA(n_edges), 
                            <np.int32_t *> np.PyArray_DATA(n_faces), 
                            n_edge_idx, n_face_idx)

        n_face_idx += 2
        n_edge_idx += 6
        
        self._insert_square(self._chalfedges[self._cfaces[face0].halfedge].prev, 
                            self._chalfedges[self._cfaces[face1].halfedge].next,  
                            <np.int32_t *> np.PyArray_DATA(n_edges), 
                            <np.int32_t *> np.PyArray_DATA(n_faces), 
                            n_edge_idx, n_face_idx)

        n_face_idx += 2
        n_edge_idx += 6

        self._insert_square(self._chalfedges[self._cfaces[face0].halfedge].next, 
                            self._chalfedges[self._cfaces[face1].halfedge].prev,  
                            <np.int32_t *> np.PyArray_DATA(n_edges), 
                            <np.int32_t *> np.PyArray_DATA(n_faces), 
                            n_edge_idx, n_face_idx)

        # Stitch the 3 squares
        self._chalfedges[n_edges[2]].twin = n_edges[11]
        self._chalfedges[n_edges[11]].twin = n_edges[2]

        self._chalfedges[n_edges[5]].twin = n_edges[14]
        self._chalfedges[n_edges[14]].twin = n_edges[5]

        self._chalfedges[n_edges[8]].twin = n_edges[17]
        self._chalfedges[n_edges[17]].twin = n_edges[8]

        # Eliminate the original faces
        self._face_delete(face0)
        self._face_delete(face1)

        # Make sure we re-calculate
        self._clear_flags()
        self.face_normals
        self.vertex_neighbors

    cdef _insert_square(self, np.int32_t edge0, np.int32_t edge1, 
                    np.int32_t * new_edges,
                    np.int32_t * new_faces,
                    int n_edge_idx,
                    int n_face_idx):
        """
        Insert a free-floating 2D square in between two mesh edges.
        """

        # Set up the first three edges as one face
        self._cfaces[new_faces[n_face_idx]].halfedge = new_edges[n_edge_idx]
        
        self._chalfedges[new_edges[n_edge_idx]].vertex = self._chalfedges[edge0].vertex
        self._chalfedges[new_edges[n_edge_idx]].face = new_faces[n_face_idx]
        if self._chalfedges[edge0].twin != -1:
            self._chalfedges[new_edges[n_edge_idx]].twin = self._chalfedges[edge0].twin
            self._chalfedges[self._chalfedges[edge0].twin].twin = new_edges[n_edge_idx]
        self._chalfedges[new_edges[n_edge_idx]].next = new_edges[n_edge_idx+1]
        self._chalfedges[new_edges[n_edge_idx]].prev = new_edges[n_edge_idx+2]

        self._chalfedges[new_edges[n_edge_idx+1]].vertex = self._chalfedges[edge1].vertex
        self._chalfedges[new_edges[n_edge_idx+1]].face = new_faces[n_face_idx]
        self._chalfedges[new_edges[n_edge_idx+1]].twin = new_edges[n_edge_idx+4]
        self._chalfedges[new_edges[n_edge_idx+1]].next = new_edges[n_edge_idx+2]
        self._chalfedges[new_edges[n_edge_idx+1]].prev = new_edges[n_edge_idx]

        self._chalfedges[new_edges[n_edge_idx+2]].vertex = self._chalfedges[self._chalfedges[edge0].prev].vertex
        self._chalfedges[new_edges[n_edge_idx+2]].face = new_faces[n_face_idx]
        self._chalfedges[new_edges[n_edge_idx+2]].twin = -1
        self._chalfedges[new_edges[n_edge_idx+2]].next = new_edges[n_edge_idx]
        self._chalfedges[new_edges[n_edge_idx+2]].prev = new_edges[n_edge_idx+1]

        # And next 3...
        self._cfaces[new_faces[n_face_idx+1]].halfedge = new_edges[n_edge_idx+3]

        self._chalfedges[new_edges[n_edge_idx+3]].vertex = self._chalfedges[edge1].vertex
        self._chalfedges[new_edges[n_edge_idx+3]].face = new_faces[n_face_idx+1]
        if self._chalfedges[edge1].twin != -1:
            self._chalfedges[new_edges[n_edge_idx+3]].twin = self._chalfedges[edge1].twin
            self._chalfedges[self._chalfedges[edge1].twin].twin = new_edges[n_edge_idx+3]
        self._chalfedges[new_edges[n_edge_idx+3]].next = new_edges[n_edge_idx+4]
        self._chalfedges[new_edges[n_edge_idx+3]].prev = new_edges[n_edge_idx+5]

        self._chalfedges[new_edges[n_edge_idx+4]].vertex = self._chalfedges[edge0].vertex
        self._chalfedges[new_edges[n_edge_idx+4]].face = new_faces[n_face_idx+1]
        self._chalfedges[new_edges[n_edge_idx+4]].twin = new_edges[n_edge_idx+1]
        self._chalfedges[new_edges[n_edge_idx+4]].next = new_edges[n_edge_idx+5]
        self._chalfedges[new_edges[n_edge_idx+4]].prev = new_edges[n_edge_idx+3]

        self._chalfedges[new_edges[n_edge_idx+5]].vertex = self._chalfedges[self._chalfedges[edge1].prev].vertex
        self._chalfedges[new_edges[n_edge_idx+5]].face = new_faces[n_face_idx+1]
        self._chalfedges[new_edges[n_edge_idx+5]].twin = -1
        self._chalfedges[new_edges[n_edge_idx+5]].next = new_edges[n_edge_idx+3]
        self._chalfedges[new_edges[n_edge_idx+5]].prev = new_edges[n_edge_idx+4]

        # New halfedges for each of the vertices
        self._cvertices[self._chalfedges[edge0].vertex].halfedge = new_edges[n_edge_idx+1]
        self._cvertices[self._chalfedges[edge1].vertex].halfedge = new_edges[n_edge_idx+4]
        self._cvertices[self._chalfedges[self._chalfedges[edge0].prev].vertex].halfedge = new_edges[n_edge_idx]
        self._cvertices[self._chalfedges[self._chalfedges[edge1].prev].vertex].halfedge = new_edges[n_edge_idx+3]

    def hole_candidate_faces(self, points, eps=10.0):
        """
        Find all mesh faces that have no points within a distance eps of their center. 
        Return the index of these faces.
        """
        tree = scipy.spatial.cKDTree(points)
        dist, _ = tree.query(self._vertices['position'][self.faces].mean(1))
        
        inds = np.flatnonzero(self._faces['halfedge'] != -1)

        return inds[dist>eps] # Optionally, (mesh._mean_edge_length + eps)], but this seems to work worse

    def pair_candidate_faces(self, candidates):
        """
        For each face, find the opposing face with the nearest centroid that has a
        normal in the opposite direction of this face and form a pair. Note this pair
        does not need to be unique.
        """
        candidate_faces = self._faces[candidates]
        candidate_halfedges = candidate_faces['halfedge']
        
        v0 = self._halfedges['vertex'][self._halfedges['prev'][candidate_halfedges]]
        v1 = self._halfedges['vertex'][candidate_halfedges]
        v2 = self._halfedges['vertex'][self._halfedges['next'][candidate_halfedges]]
        
        candidate_vertices = np.vstack([v0,v1,v2]).T
        candidate_positions = self._vertices['position'][candidate_vertices] # (N, (v0,v1,v2), (x,y,z))
        candidate_centroids = candidate_positions.mean(1)
        
        candidate_normals = candidate_faces['normal']  # (N, 3)
        
        # Compute the shift orthogonal to the mean normal plane between each of the faces
        candidate_shift = candidate_centroids[None,...] - candidate_centroids[:,None,:]  # (N, N, 3)
        n_hat = 0.5*(candidate_normals[None,...] + candidate_normals[:,None,:])  # (N, N, 3)
        shift = candidate_shift - n_hat*(((n_hat*candidate_shift).sum(2))[...,None])
        abs_shift = (shift*shift).sum(2)
        
        # Compute the dot product between all of the normals
        nd = (candidate_normals[None,...]*candidate_normals[:,None,:]).sum(2)  # (N, N)
        
        # For each face, find the opposing face with the nearest centroid that has a
        # normal in the opposite direction of this face
        factor = -0.5
        ndlt = nd<factor  # TODO: stricter requirement on "opposing face" angle?
        min_mask = np.any(ndlt,axis=1)  
        min_inds = np.argmax(-abs_shift*ndlt-1e6*(nd>=factor),axis=1)
        pairs = np.vstack([np.flatnonzero(min_mask), min_inds[min_mask]])
        
        return candidates[pairs[0,:]], pairs[1,:]

    def empty_prism_candidate_faces(self, points, candidates, candidate_pair, eps=10.0):
        """
        For each candidate pair, check that there are no points in between the candidate triangles.
        This expects candidate, candidate_pair output from pair_candidate_faces(), where the
        candidates are face indices in mesh, and candidate_pair is an index into candidates,
        indicating the face paired with candidate i for i \in range(len(candidates)).
        """
        tree = scipy.spatial.cKDTree(points)
        kept_cands = np.zeros_like(candidates, dtype=bool)
        disallowed = np.zeros_like(candidates, dtype=bool)
        
        # face centers
        he = self._faces['halfedge'][candidates]
        v0 = self._halfedges['vertex'][self._halfedges['prev'][he]]
        v1 = self._halfedges['vertex'][he]
        v2 = self._halfedges['vertex'][self._halfedges['next'][he]]
        fv = np.vstack([v0,v1,v2]).T
        fv_pos = self._vertices['position'][fv]
        face_centers = fv_pos.mean(1)
        
        # half-planes
        n = self._faces['normal'][candidates]
        v01 = fv_pos[:,0]-fv_pos[:,1]
        v12 = fv_pos[:,1]-fv_pos[:,2]
        v20 = fv_pos[:,2]-fv_pos[:,0]
        hp0 = np.cross(n, v01, axis=1)/np.linalg.norm(v01,axis=1)[:,None]
        hp1 = np.cross(n, v12, axis=1)/np.linalg.norm(v12,axis=1)[:,None]
        hp2 = np.cross(n, v20, axis=1)/np.linalg.norm(v20,axis=1)[:,None]
        
        for i, ci in enumerate(candidates):
            # Find the points within ||t_i-t_j||+eps of t_i or t_j, where t_{i,j} are pair triangle centroids
            j = candidate_pair[i]
            
            if kept_cands[i] or disallowed[i] or kept_cands[j] or disallowed[j]:
                # This candidate is already paired (disallow repeats)
                continue
            
            fci, fcj = face_centers[i], face_centers[j]
            r = np.sqrt(((fci-fcj)*(fci-fcj)).sum())+eps
            p = tree.query_ball_point([fci,fcj], r)
            # p = np.hstack([p[0],p[1]]).ravel()
            
            p = np.array([y for x in p for y in x if len(x) > 0])

            if len(p) == 0:
                # There are no points within r of either of these faces
                kept_cands[i] |= False
                # disallowed[candidates == candidates[j]] |= True
                continue

            # Check if any of these are within +eps of half planes of both triangles
            # A triangle half-plane is the plane spanned by a triangle edge and the triangle normal
            below_hp0_ci = (hp0[i][None,:]*(points[p]-fv_pos[i,1][None,:])).sum(1) < eps
            below_hp1_ci = (hp1[i][None,:]*(points[p]-fv_pos[i,2][None,:])).sum(1) < eps
            below_hp2_ci = (hp2[i][None,:]*(points[p]-fv_pos[i,0][None,:])).sum(1) < eps
            
            below_hp0_cj = (hp0[j][None,:]*(points[p]-fv_pos[j,1][None,:])).sum(1) < eps
            below_hp1_cj = (hp1[j][None,:]*(points[p]-fv_pos[j,2][None,:])).sum(1) < eps
            below_hp2_cj = (hp2[j][None,:]*(points[p]-fv_pos[j,0][None,:])).sum(1) < eps
            
            # If no points are in between these triangles, keep them
            inside = np.sum(below_hp0_ci & below_hp1_ci & below_hp2_ci \
                            & below_hp0_cj & below_hp1_cj & below_hp2_cj) == 0
            
            kept_cands[i] |= inside
            disallowed[candidates == candidates[j]] |= inside
            
        c = candidates[kept_cands]
        cp = candidates[candidate_pair[kept_cands]]
        
        return np.hstack([c,cp]), np.hstack([range(len(c),2*len(c)), range(len(c))])

    def connect_candidates(self, candidates):
        """
        Compute the connected component labeling of the kept faces 
        such that faces that share edges are considered connected.
        """
        # mesh._components_valid = 0
        self._faces['component'][:] = 1e6  # -1 out the componenets
        
        # Give each face its own component
        self._faces['component'][candidates] = range(len(candidates))
        
        for c in candidates:
            # Assign each face the minimum component of it and the face of its twin edge
            e0 = self._faces['halfedge'][c]
            e1 = self._halfedges['next'][e0]
            e2 = self._halfedges['prev'][e0]
        
            c0, c1, c2 = 1e6, 1e6, 1e6
            if self._halfedges['twin'][e0] != -1:
                c0 = self._faces['component'][self._halfedges['face'][self._halfedges['twin'][e0]]]
            if self._halfedges['twin'][e1] != -1:
                c1 = self._faces['component'][self._halfedges['face'][self._halfedges['twin'][e1]]]
            if self._halfedges['twin'][e2] != -1:
                c2 = self._faces['component'][self._halfedges['face'][self._halfedges['twin'][e2]]]

            new_component = np.min([self._faces['component'][c], c0, c1, c2])
            
            self._faces['component'][c] = new_component
            if self._halfedges['face'][self._halfedges['twin'][e0]] in candidates:
                self._faces['component'][self._halfedges['face'][self._halfedges['twin'][e0]]] = new_component
            if self._halfedges['face'][self._halfedges['twin'][e1]] in candidates:
                self._faces['component'][self._halfedges['face'][self._halfedges['twin'][e1]]] = new_component
            if self._halfedges['face'][self._halfedges['twin'][e2]] in candidates:
                self._faces['component'][self._halfedges['face'][self._halfedges['twin'][e2]]] = new_component
            
        return self._faces['component'][candidates]

    def connected_candidates_euler_characteristic(self, candidates, component):
        unique_components = np.unique(component)
        chi = np.zeros_like(unique_components)
        for i, c in enumerate(unique_components):
            he = self._faces['halfedge'][candidates[component==c]]
            v0 = self._halfedges['vertex'][self._halfedges['prev'][he]]
            v1 = self._halfedges['vertex'][he]
            v2 = self._halfedges['vertex'][self._halfedges['next'][he]]
            fv = np.hstack([v0,v1,v2])
            
            F = len(he)
            V = len(set(fv.ravel()))
            
            # grab all edges
            edges = np.vstack([fv, np.hstack([v1,v2,v0])]).T

            # Sort lo->hi
            sorted_edges = np.sort(edges, axis=1)

            # find the number of unique elements
            E = len(np.unique(sorted_edges,axis=0))
            
            chi[i] = V - E + F
        
        return chi

    def update_topology(self, candidates, candidate_pairs, component, euler):
        unique_components = np.unique(component)
        used_components = np.zeros_like(unique_components, dtype=bool)
        for i, c in enumerate(unique_components):
            if used_components[i]:
                # We've already used this in a different hole punch
                continue
            component_idxs = component==c
            component_cands = candidates[component_idxs]
            if euler[i] == 0:
                # TODO: Currently disabled due to problems with repair() after _face_delete

                # This is topologically a cylinder
                
                # 1. Delete all faces in this component
                # for face in component_cands:
                #     self._face_delete(face)
                    
                # 2. Patch holes in the resulting boundaries
                # self.repair()
                pass
                
            elif euler[i] == 1:
                # This is topologically a plane. If there is a pair in another
                # component, punch a hole and remove both components from consideration.
                component_cand_pairs = candidate_pairs[component_idxs]
                for j, pair_idx in enumerate(component_cand_pairs):
                    pair_component_idx = np.flatnonzero(unique_components==component[pair_idx])
                    assert(len(pair_component_idx) == 1)
                    pair_component_idx = [0]
                    if (component[pair_idx] != c) and (used_components[pair_component_idx] != True):
                        self._punch_hole(component_cands[j], candidates[pair_idx])
                        used_components[pair_component_idx] = True
                        break
            else:
                print(f"Component {c} has Euler characteristic {euler[i]}. I don't know what to do with this.")
            # Mark this component as used
            used_components[i] = True

    def cut_and_punch(self, pts, eps=10.0):
        # Find all mesh faces that have no points within eps of their face center
        hc = self.hole_candidate_faces(pts, eps=eps/5.0)  # TODO: 5.0 is empirical

        # Pair these faces by matching each face to its closest face in mean normal space
        # with an opposing normal. Allows many-to-one.
        cands, pairs = self.pair_candidate_faces(hc)

        # Check if there are no points within eps of the prism formed by each face pair. Keep these
        # only. Restores one-to-one face matching.
        empty_cands, empty_pairs = self.empty_prism_candidate_faces(pts, cands, pairs, eps=eps)

        # Group the remaining faces by edge connectivity.
        component = self.connect_candidates(empty_cands)

        # Compute the euler characteristic of each component. Euler 0 = tube, 1 = plane patch.
        chi = self.connected_candidates_euler_characteristic(empty_cands, component)

        # Punch holes between place patches (cut tubes is currently disabled)
        self.update_topology(empty_cands, empty_pairs, component, chi)

    cdef grad(self, points, sigma):
        """
        Gradient between points and the surface.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit.
            sigma : float
                Localization uncertainty of points.
        """
        # attraction = np.zeros((self._vertices.shape[0], 3), dtype=np.float32)
        # cdef points_t[:] attraction_view = attraction.ravel().view(POINTS_DTYPE)
        # cdef points_t[:] points_view = points.ravel().view(POINTS_DTYPE)
        # cdef float[:] sigma_view = sigma

        dN = 0.1
        if USE_C:
            curvature = self.curvature_grad_c(dN=dN, skip_prob=self.skip_prob)
        else:
            curvature = self.curvature_grad(dN=dN, skip_prob=self.skip_prob)
        attraction = self.point_attraction_grad_kdtree(points, sigma, w=0.95, search_k=self.search_k)
        # c_point_attraction_grad(&(attraction_view[0]), 
        #                        &(points_view[0]), 
        #                        &(sigma_view[0]), 
        #                        &(self._cvertices[0]), 
        #                        0.95, 
        #                        self._mean_edge_length/2.5, 
        #                        points.shape[0], 
        #                        self._vertices.shape[0])

        # ratio = np.nanmean(np.linalg.norm(curvature,axis=1)/np.linalg.norm(attraction,axis=1))
        # print('Ratio: ' + str(ratio))

        # c_inf_mask = (np.isinf(curvature).sum(1)>0)
        # a_inf_mask = (np.isinf(attraction).sum(1)>0)

        # c_inf = curvature[c_inf_mask]
        # a_inf = attraction[a_inf_mask]

        # if len(c_inf) > 0:
        #     print('Curvature infinity!!!')
        #     print(self._vertices[c_inf_mask])

        # if len(a_inf) > 0:
        #     print('Attraction infinity!!!')
        #     print(self._vertices[a_inf_mask])

        print("Curvature: {}".format(np.mean(curvature,axis=0)))
        print("Attraction: {}".format(np.mean(attraction,axis=0)))
        print("Curvature-to-attraction: {}".format(np.mean(curvature/attraction,axis=0)))

        g = self.a*attraction + self.c*curvature
        print("Gradient: {}".format(np.mean(g,axis=0)))
        return g

    def opt_adam(self, points, sigma, max_iter=250, step_size=1, beta_1=0.9, beta_2=0.999, eps=1e-8, **kwargs):
        """
        Performs Adam optimization (https://arxiv.org/abs/1412.6980) on
        fit of surface mesh surf to point cloud points.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit.
            sigma : float
                Localization uncertainty of points.
        """
        # Initialize moment vectors
        m = np.zeros(self._vertices['position'].shape)
        v = np.zeros(self._vertices['position'].shape)

        t = 0
        # g_mag_prev = 0
        # g_mag = 0
        while (t < max_iter):
            print('Iteration %d ...' % t)
            
            t += 1
            # Gaussian noise std
            noise_sigma = np.sqrt(self.step_size / ((1 + t)**0.55))
            # Gaussian noise
            noise = np.random.normal(0, noise_sigma, self._vertices['position'].shape)
            # Calculate graident for each point on the  surface, 
            g = self.grad(points, sigma)
            # add Gaussian noise to the gradient
            g += noise
            # Update first biased moment 
            m = beta_1 * m + (1. - beta_1) * g
            # Update second biased moment
            v = beta_2 * v + (1. - beta_2) * np.multiply(g, g)
            # Remove biases on moments & calculate update weight
            a = step_size * np.sqrt(1. - beta_2**t) / (1. - beta_1**t)
            # Update the surface
            self._vertices['position'] += a * m / (np.sqrt(v) + eps)

    def opt_euler(self, points, sigma, max_iter=100, step_size=1, eps=0.00001, **kwargs):
        """
        Normal gradient descent.

        Parameters
        ----------
            points : np.array
                3D point cloud to fit.
            sigma : float
                Localization uncertainty of points.
        """

        # Precalc 
        dr = (self.delaunay_remesh_frequency != 0)
        r = (self.remesh_frequency != 0)
        if r:
            initial_length = self._mean_edge_length
            final_length = 3*np.max(sigma)
            m = (final_length - initial_length)/max_iter
        
        for _i in np.arange(max_iter):

            print('Iteration %d ...' % _i)
            
            # Calculate the weighted gradient
            shift = step_size*self.grad(points, sigma)

            # Update the vertices
            self._vertices['position'] += shift

            # self._faces['normal'][:] = -1
            # self._vertices['neighbors'][:] = -1
            self._face_normals_valid = 0
            self._vertex_normals_valid = 0
            self.face_normals
            self.vertex_neighbors

            # If we've reached precision, terminate
            if np.all(shift < eps):
               break

            if (_i == 0):
                # Don't remesh
                continue

            # Remesh
            if r and ((_i % self.remesh_frequency) == 0):
                target_length = initial_length + m*_i
                self.remesh(5, target_length, 0.5, 10)
                print('Target mean length: {}   Resulting mean length: {}'.format(str(target_length), 
                                                                                str(self._mean_edge_length)))

            # Delaunay remesh
            if dr and ((_i % self.delaunay_remesh_frequency) == 0):
                self.delaunay_remesh(points, self.delaunay_eps)

    def opt_expectation_maximization(self, points, sigma, max_iter=100, step_size=1, eps=0.00001, **kwargs):
        for _i in np.arange(max_iter):

            print('Iteration %d ...' % _i)

            if _i % 2:
                dN = 0.1
                # grad = self.c*self.curvature_grad(dN=dN)
                if USE_C:
                    grad = self.curvature_grad_c(dN=dN, skip_prob=self.skip_prob)
                else:
                    grad = self.curvature_grad(dN=dN, skip_prob=self.skip_prob)
            else:
                grad = self.a*self.point_attraction_grad_kdtree(points, sigma)

            # Calculate the weighted gradient
            shift = step_size*grad

            # Update the vertices
            self._vertices['position'] += shift

            # self._faces['normal'][:] = -1
            # self._vertices['neighbors'][:] = -1
            self._face_normals_valid = 0
            self._vertex_normals_valid = 0
            self.face_normals
            self.vertex_neighbors

            # If we've reached precision, terminate
            if np.all(shift < eps):
                return

    def opt_conjugate_gradient(self, points, sigma, max_iter=10, step_size=1.0, **kwargs):
        from ch_shrinkwrap.conj_grad import ShrinkwrapConjGrad

        r = (self.remesh_frequency != 0) and (self.remesh_frequency <= max_iter)
        dr = (self.delaunay_remesh_frequency != 0) and (self.delaunay_remesh_frequency <= max_iter)

        if r and dr:
            # Make sure we stop for both
            rf = math.gcd(self.remesh_frequency, self.delaunay_remesh_frequency)
        elif r:
            rf = self.remesh_frequency
        elif dr:
            rf = self.delaunay_remesh_frequency
        else:
            rf = max_iter

        if r:
            initial_length = self._mean_edge_length
            if kwargs.get('minimum_edge_length') < 0:
                final_length = np.clip(np.min(sigma)/2.5, 1.0, 50.0)
            else:
                final_length = kwargs.get('minimum_edge_length')
            m = (final_length - initial_length)/max_iter


        if (len(sigma.shape) == 1) and (sigma.shape[0] == points.shape[0]):
            print("Not this case???")
            print(points.shape, sigma.shape)
            s = 1.0/np.repeat(sigma,points.shape[1])
            print(points.shape, sigma.shape)
        elif (len(sigma.shape) == 2) and (sigma.shape[0] == points.shape[0]) and (sigma.shape[1] == points.shape[1]):
            print('this case')
            print(points.shape, sigma.shape)
            s = (1.0/sigma.ravel())
            print(points.shape, sigma.shape)
        else:
            raise ValueError(f"Sigma must be of shape ({self.points.shape[0]},) or ({self.points.shape[0]},{self.points.shape[1]}).")

        # initialize area values (used in termination condition)
        original_area = self.area()
        last_area, area = original_area, 0

        self.cg = None

        j = 0

        while j < max_iter:
            n = self._halfedges['vertex'][self._vertices['neighbors']]
            n_idxs = self._vertices['neighbors'] == -1
            n[n_idxs] = -1
            fn = self._halfedges['face'][self._vertices['neighbors']]
            fn[n_idxs] = -1
            faces = self._faces['halfedge']
            v0 = self._halfedges['vertex'][self._halfedges['prev'][faces]]
            v1 = self._halfedges['vertex'][faces]
            v2 = self._halfedges['vertex'][self._halfedges['next'][faces]]
            faces_by_vertex = np.vstack([v0, v1, v2]).T
            self.cg = ShrinkwrapConjGrad(self._vertices['position'], n, faces_by_vertex, fn, points, 
                                    search_k=self.search_k, search_rad=self.search_rad,
                                    shield_sigma=self._mean_edge_length/2.0)

            n_it = min(max_iter - j, rf)
            vp = self.cg.search(points,lams=step_size*self.kc/2.0,num_iters=n_it,
                           weights=s)

            j += n_it

            k = (self._vertices['halfedge'] != -1)
            self._vertices['position'][k] = vp[k]

            # self._faces['normal'][:] = -1
            # self._vertices['neighbors'][:] = -1
            self._face_normals_valid = 0
            self._vertex_normals_valid = 0
            self.face_normals
            self.vertex_neighbors

            # Delaunay remesh (hole punch)
            if dr and ((j % self.delaunay_remesh_frequency) == 0):
                self.delaunay_remesh(points, self.delaunay_eps)
                break

            # Remesh
            if r and ((j % self.remesh_frequency) == 0):
                target_length = initial_length + m*(j+1)
                # target_length = np.maximum(0.5*self._mean_edge_length, final_length)
                self.remesh(5, target_length, 0.5, 10)
                print('Target mean length: {}   Resulting mean length: {}'.format(str(target_length), 
                                                                                str(self._mean_edge_length)))             
                self.cg = None

            # Terminate if area change is minimal
            area = self.area()
            area_ratio = math.fabs(last_area-area)/last_area
            print(f"Area ratio is {area_ratio:.4f}")
            # if area_ratio < 0.001:
            #     print(f"CONVERGED in {j*rf}!!!")
            #     break
            last_area = area

    # make some metrics from the optimiser accessible for visualisation
    @property
    def _S0(self):
        """ Search direction to minimise data misfit"""
        return self.cg.Ahfunc(self.cg.res).reshape(self.vertices.shape)

    @property
    def point_dis(self):
        """
        As we can't plot the vectorial search direction, this is the magnitude of it

        TODO - make the naming better
        """
        s0 = self._S0
        return np.sqrt((s0*s0).sum(1))

    @property
    def rms_point_sc(self):
        """
        An attempt at characterising the how much point-error lies beneath a given vertex, and avoiding
        situations where points pulling in opposite direction cancel by taking their magnitudes first.

        In practice seems to do a better job of measuring Ahfunc (ie how strongly a given vertex is
        constrained by the membrane). Suspect we need to do this without some of the normalisations we
        currently have.
        """
        rn = np.sqrt((self.cg.res*self.cg.res).reshape(self.cg.points.shape).sum(1))[:,None]*np.ones(3)[None,:].ravel()
        rme = self.cg.Ahfunc(rn).reshape(self.vertices.shape)
        return np.sqrt((rme*rme).sum(1))

    #@property
    #def _S1(self):
    #    """ Search direction to minimise regularisation term (curvature)""""
    #    raise NotImplementedError('this should mirror S1 in subseearch')

    def opt_skeleton(self, points, sigma, max_iter=10, lam=[0], target_edge_length=-1, **kwargs):
        from ch_shrinkwrap.conj_grad import SkeletonConjGrad

        self.remesh_frequency = 1
        rf = 1  # manually set remesh frequency to 1

        # initialize area values (used in termination condition)
        original_area = self.area()
        last_area, area = original_area, 0
        area_variation_factor = kwargs.get('area_variation_factor')
        max_triangle_angle = PI*kwargs.get('max_triangle_angle')/180.0

        # initialize skeleton, construct Voronoi diagram once
        n = self._halfedges['vertex'][self._vertices['neighbors']]
        n[self._vertices['neighbors'] == -1] = -1
        cg = SkeletonConjGrad(self._vertices['position'], self._vertices['normal'], n, mesh=self)
        
        # Loop over iterations
        for j in range(max_iter//rf):
            k = (self._vertices['halfedge'] != -1)
            print(f"{k.sum()} vertices")

            n = self._halfedges['vertex'][self._vertices['neighbors']]
            n[self._vertices['neighbors'] == -1] = -1

            # Update positions
            cg.vertex_neighbors, cg.vertex_normals, cg.vertices = n, self._vertices['normal'], self._vertices['position']

            vp = cg.search(np.zeros_like(self._vertices['position']),lams=lam,num_iters=rf)

            self._vertices['position'][k] = vp[k]

            # self._faces['normal'][:] = -1
            # self._vertices['neighbors'][:] = -1
            self._face_normals_valid = 0
            self._vertex_normals_valid = 0
            self.face_normals
            self.vertex_neighbors

            # Remesh
            if ((((j+1)*rf) % self.remesh_frequency) == 0):
                # print(f"Max angle: {max_triangle_angle}")
                # self.skeleton_remesh(target_edge_length=target_edge_length,
                #                      max_triangle_angle=max_triangle_angle)
                self.remesh(target_edge_length=target_edge_length)

            # Check if area ratio is met
            area = self.area()
            area_ratio = math.fabs(last_area-area)/original_area
            print(f"Area ratio is {area_ratio:.4f}")
            if area_ratio < area_variation_factor:
                break
            last_area = area
        
    def shrink_wrap(self, points, sigma, method='conjugate_gradient', max_iter=None, **kwargs):

        if method not in DESCENT_METHODS:
            print('Unknown gradient descent method. Using {}.'.format(DEFAULT_DESCENT_METHOD))
            method = DEFAULT_DESCENT_METHOD

        if max_iter is None:
            max_iter = self.max_iter
        
        opts = dict(points=points,
                    sigma=sigma, 
                    max_iter=max_iter,
                    step_size=self.step_size, 
                    beta_1=self.beta_1, 
                    beta_2=self.beta_2,
                    eps=self.eps,
                    **kwargs)

        return getattr(self, 'opt_{}'.format(method))(**opts)

        # if method == 'euler':
        #     return self.opt_euler(**opts)
        # elif method == 'expectation_maximization':
        #     return self.opt_expectation_maximization(**opts)
        # elif method == 'adam':
        #     return self.opt_adam(**opts)

cimport numpy as np
import numpy as np
import scipy.spatial
import cython
import math

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
    cdef object _dH
    cdef object _dK
    cdef object _dE_neighbors
    cdef float * _ck_0
    cdef float * _ck_1
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
        
        self.vertex_properties.extend(['E', 'curvature_principal0', 'curvature_principal1']) #, 'puncture_candidates'])

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
        if self._E is None:
            self._populate_curvature_grad()
            self._E[np.isnan(self._E)] = 0
        return self._E

    @property
    def pE(self):
        if self._pE is None:
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
        self._pE = np.zeros(sz, dtype=np.float32)
        self._dH = np.zeros(sz, dtype=np.float32)
        self._dK = np.zeros(sz, dtype=np.float32)
        self._dE_neighbors = np.zeros(sz, dtype=np.float32)

        self._set_ck_0(self._k_0)
        self._set_ck_1(self._k_1)
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

    cpdef int skeleton_edge_split(self, np.int32_t _curr, bint live_update=1):
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

        # FROM HERE
        # project the _next vertex perpendicularly onto the edge
        n0x = x1x - x0x
        n0y = x1y - x0y
        n0z = x1z - x0z
        n1x = self._cvertices[self._chalfedges[_next].vertex].position0 - x0x
        n1y = self._cvertices[self._chalfedges[_next].vertex].position1 - x0y
        n1z = self._cvertices[self._chalfedges[_next].vertex].position2 - x0z

        # displacement along the x1-x0 direction
        #print(curr_edge.length)
        if curr_edge.length > 1e-6:
            ndot = ((n0x*n1x)+(n0y*n1y)+(n0z*n1z))/(curr_edge.length*curr_edge.length)
        else:
            ndot = 0.5

        _vertex[0] = x0x + (x1x-x0x)*ndot
        _vertex[1] = x0y + (x1y-x0y)*ndot
        _vertex[2] = x0z + (x1z-x0z)*ndot

        # TO HERE

        _vertex_idx = self._new_vertex(_vertex)

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
            _face_1_idx = self._new_face(_twin_prev)
            self._chalfedges[_twin_prev].face = _face_1_idx
        _face_2_idx = self._new_face(_next)
        self._chalfedges[_next].face = _face_2_idx

        # Insert the new faces
        _he_0_idx = self._insert_new_edge(self._chalfedges[_next].vertex, prev=_curr, next=_prev, face=self._chalfedges[_curr].face)
        if interior:
            _he_1_idx = self._insert_new_edge(_vertex_idx, prev=_twin_next, next=_twin, face=self._chalfedges[_twin].face)
        
            _he_2_idx = self._insert_new_edge(self._chalfedges[_twin_next].vertex, next=_twin_prev, face=_face_1_idx)
            _he_3_idx = self._insert_new_edge(_vertex_idx, prev=_twin_prev, next=_he_2_idx, face=_face_1_idx)
            self._chalfedges[_he_2_idx].prev = _he_3_idx

        _he_4_idx = self._insert_new_edge(self._chalfedges[_curr].vertex, next=_next, face=_face_2_idx, twin=-1)
        _he_5_idx = self._insert_new_edge(_vertex_idx, prev=_next, next=_he_4_idx, face=_face_2_idx)
        self._chalfedges[_he_4_idx].prev = _he_5_idx

        self._chalfedges[_he_0_idx].twin = _he_5_idx
        self._chalfedges[_he_5_idx].twin = _he_0_idx

        if interior:
            self._chalfedges[_he_1_idx].twin = _he_2_idx
            self._chalfedges[_he_2_idx].twin = _he_1_idx

            self._chalfedges[_he_3_idx].twin = _he_4_idx
            self._chalfedges[_he_4_idx].twin = _he_3_idx

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

        if live_update:
            if interior:
                #self._update_face_normals([self._chalfedges[_he_0_idx].face, self._chalfedges[_he_1_idx].face, self._chalfedges[_he_2_idx].face, self._chalfedges[_he_4_idx].face])
                #self._update_vertex_neighbors([self._chalfedges[_curr].vertex, self._chalfedges[_twin].vertex, self._chalfedges[_he_0_idx].vertex, self._chalfedges[_he_2_idx].vertex, self._chalfedges[_he_4_idx].vertex])
            
                update_face_normal(self._chalfedges[_he_0_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_1_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_2_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_4_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                
                update_single_vertex_neighbours(self._chalfedges[_curr].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_twin].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_he_0_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_he_2_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_he_4_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
            
            else:
                #self._update_face_normals([self._chalfedges[_he_0_idx].face, self._chalfedges[_he_4_idx].face])
                #self._update_vertex_neighbors([self._chalfedges[_curr].vertex, self._chalfedges[_prev].vertex, self._chalfedges[_he_0_idx].vertex, self._chalfedges[_he_4_idx].vertex])
                
                update_face_normal(self._chalfedges[_he_0_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                update_face_normal(self._chalfedges[_he_4_idx].face, self._chalfedges, self._cvertices, self._cfaces)
                
                update_single_vertex_neighbours(self._chalfedges[_curr].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_prev].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_he_0_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(self._chalfedges[_he_4_idx].vertex, self._chalfedges, self._cvertices, self._cfaces)
            
            
            self._faces_by_vertex = None
            self._H = None
            self._K = None
        
        return 1

    cdef int skeleton_split_edges(self):
        cdef int split_count = 0
        cdef int i
        cdef int n_halfedges = self._halfedges.shape[0]
        cdef float *v0
        cdef float *v1
        cdef float *v2
        cdef float *v3
        cdef float *v4
        cdef float *v5
        cdef float t0d
        cdef float t1d
        cdef float t0l
        cdef float t1l
        
        for i in range(n_halfedges):
            if (self._chalfedges[i].vertex != -1):
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

                if (t0d < COS110) and (t0d < t1d):
                    #print(f"Splitting {i}")
                    self.skeleton_edge_split(i)
                    split_count += 1
                elif (t1d < COS110) and (t1d < t0d):
                    #print(f"Splitting {self._chalfedges[i].twin}")
                    self.skeleton_edge_split(self._chalfedges[i].twin)
                    split_count += 1
                
        print('Split count: %d' % (split_count))
        return split_count

    cdef int skeleton_remesh(self, float target_edge_length=-1):
        cdef int k, i, ct
        cdef float collapse_threshold, xl, xu, yl, yu, zl, zu, diag

        cdef int n_halfedges = self._halfedges.shape[0]
        cdef int n_vertices = self._vertices.shape[0]

        if (target_edge_length < 0):
            # Guess edge_length
            xl, yl, zl, xu, yu, zu = self.bbox
            diag = math.sqrt((xu-xl)*(xu-xl)+(yu-yl)*(yu-yl)+(zu-zl)*(zu-zl))
            collapse_threshold = 0.002*diag
        else:
            collapse_threshold = target_edge_length
        
        print(f"Target edge length: {collapse_threshold}")
        ct = self.collapse_edges(collapse_threshold)

        ct = self.skeleton_split_edges()

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
        ext_inds = delaunay_utils.ext_simps(tri, self)
        simps = delaunay_utils.del_simps(tri, ext_inds)

        # Remove simplices that do not contain points
        # eps = self._mean_edge_length/5.0  # How far outside of a tetrahedron do we 
                                          # consider a point 'inside' a tetrahedron?
                                          # TODO: /5.0 is empirical. sqrt(6)/4*base length is circumradius
                                          # TODO: account for sigma?
        #print('Guessed eps: {}'.format(eps))
        empty_inds = delaunay_utils.empty_simps(simps, v, points, eps=eps)
        simps_ = delaunay_utils.del_simps(simps, empty_inds)

        # Recover new triangulation
        faces = delaunay_utils.surf_from_delaunay(simps_)

        # Rebuild mesh
        self.build_from_verts_faces(v, faces, clear=True)

        # Delaunay remeshing has a penchant for flanges
        # self._remove_singularities()
        self.repair()

        self._initialize_curvature_vectors()

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
            from math import gcd
            rf = gcd(self.remesh_frequency, self.delaunay_remesh_frequency)
        elif r:
            rf = self.remesh_frequency
        elif dr:
            rf = self.delaunay_remesh_frequency
        else:
            rf = max_iter

        if r:
            initial_length = self._mean_edge_length
            final_length = np.clip(np.min(sigma)/2.5, 1.0, 50.0)
            m = (final_length - initial_length)/max_iter

        # initialize area values (used in termination condition)
        original_area = self.area()
        last_area, area = original_area, 0

        for j in range(max_iter//rf):
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
            cg = ShrinkwrapConjGrad(self._vertices['position'], n, faces_by_vertex, fn, points, 
                                    search_k=self.search_k, search_rad=self.search_rad)

            vp = cg.search(points,lams=step_size,num_iters=rf,
                           weights=1.0/np.repeat(sigma,points.shape[1]))

            k = (self._vertices['halfedge'] != -1)
            self._vertices['position'][k] = vp[k]

            # self._faces['normal'][:] = -1
            # self._vertices['neighbors'][:] = -1
            self._face_normals_valid = 0
            self._vertex_normals_valid = 0
            self.face_normals
            self.vertex_neighbors

            # Delaunay remesh (hole punch)
            if dr and ((((j+1)*rf) % self.delaunay_remesh_frequency) == 0):
                self.delaunay_remesh(points, self.delaunay_eps)

            # Remesh
            if r and ((((j+1)*rf) % self.remesh_frequency) == 0):
                target_length = initial_length + m*(j+1)*rf
                # target_length = np.maximum(0.5*self._mean_edge_length, final_length)
                self.remesh(5, target_length, 0.5, 10)
                print('Target mean length: {}   Resulting mean length: {}'.format(str(target_length), 
                                                                                str(self._mean_edge_length)))

            # Terminate if area change is minimal
            area = self.area()
            area_ratio = math.fabs(last_area-area)/last_area
            print(f"Area ratio is {area_ratio:.4f}")
            # if area_ratio < 0.001:
            #     print(f"CONVERGED in {j*rf}!!!")
            #     break
            last_area = area

    def opt_skeleton(self, points, sigma, max_iter=10, lam=[0,0], target_edge_length=-1, **kwargs):
        from ch_shrinkwrap.conj_grad import SkeletonConjGrad

        self.remesh_frequency = 1

        r = (self.remesh_frequency != 0) and (self.remesh_frequency < max_iter)

        if r:
            rf = self.remesh_frequency
        else:
            rf = max_iter

        # initialize area values (used in termination condition)
        original_area = self.area()
        last_area, area = original_area, 0

        # initialize skeleton, construct Voronoi diagram once
        n = self._halfedges['vertex'][self._vertices['neighbors']]
        n[self._vertices['neighbors'] == -1] = -1
        cg = SkeletonConjGrad(self._vertices['position'], self._vertices['normal'], n, mesh=self)
        for j in range(max_iter//rf):
            k = (self._vertices['halfedge'] != -1)
            print(f"{k.sum()} vertices")

            n = self._halfedges['vertex'][self._vertices['neighbors']]
            n[self._vertices['neighbors'] == -1] = -1

            # Update positions
            cg.neighbors, cg.vertex_normals, cg.vertices = n, self._vertices['normal'], self._vertices['position']

            vp = cg.search(np.zeros_like(self._vertices['position']),lams=lam,num_iters=rf)

            self._vertices['position'][k] = vp[k]

            # self._faces['normal'][:] = -1
            # self._vertices['neighbors'][:] = -1
            self._face_normals_valid = 0
            self._vertex_normals_valid = 0
            self.face_normals
            self.vertex_neighbors

            # Check if area ratio is met
            area = self.area()
            area_ratio = math.fabs(last_area-area)/original_area
            print(f"Area ratio is {area_ratio:.4f}")
            if area_ratio < 0.0001:
                break
            last_area = area

            # Remesh
            if r and ((((j+1)*rf) % self.remesh_frequency) == 0):
                self.skeleton_remesh(target_edge_length=target_edge_length)
        
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

cimport numpy as np
import numpy as np
import scipy.spatial
import cython
import math
import ctypes

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from PYME.experimental._triangle_mesh cimport TriangleMesh
from PYME.experimental._triangle_mesh import TriangleMesh
from PYME.experimental._triangle_mesh import VERTEX_DTYPE

from ch_shrinkwrap import membrane_mesh_utils
from ch_shrinkwrap import delaunay_utils

# Gradient descent methods
DESCENT_METHODS = ['conjugate_gradient', 'skeleton']
DEFAULT_DESCENT_METHOD = 'conjugate_gradient'

DEF KBT = 0.0257  # eV # 4.11e-21  # joules
DEF NM2M = 1
DEF COS110 = -0.34202014332
#cdef const float PI 3.1415927
DEF PI=3.1415927

MAX_VERTEX_COUNT = 2**31

I = np.eye(3, dtype=float)

DEF USE_C = True

POINTS_DTYPE = np.dtype([('position', '3f4')])
POINTS_DTYPE2 = np.dtype([('position0', 'f4'), 
                          ('position1', 'f4'), 
                          ('position2', 'f4')])

cdef extern from "triangle_mesh_utils.c":
    void _update_face_normals(np.int32_t *f_idxs, 
                              halfedge_t *halfedges, 
                              vertex_t *vertices, 
                              face_t *faces, 
                              signed int n_idxs)
    void update_face_normal(int f_idx, halfedge_t *halfedges, vertex_d *vertices, face_d *faces)
    void update_single_vertex_neighbours(int v_idx, halfedge_t *halfedges, vertex_d *vertices, face_d *faces)

cdef extern from "membrane_mesh_utils.c":
    void fcompute_curvature_tensor_eig(float *Mvi, float *l1, float *l2, float *v1, float *v2) 
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
    void c_holepunch_pair_candidate_faces(void *vertices_, 
                                          void *faces_,
                                          halfedge_t *halfedges,
                                          int *candidates,
                                          int n_candidates,
                                          int *pairs)

cdef class MembraneMesh(TriangleMesh):
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


        self._initialize_curvature_vectors()
        
        self.vertex_properties.extend(['E', 'curvature_principal0', 'curvature_principal1', 'point_dis', 'rms_point_sc', 'point_influence']) #, 'puncture_candidates'])

        # Number of neighbors to use in self.point_attraction_grad_kdtree
        self.search_k = 200
        self.search_rad = 100

        # Percentage of vertices to skip on each refinement iteration
        self.skip_prob = 0.0

        # Pointcloud kdtree
        self._tree = None

        for key, value in kwargs.items():
            setattr(self, key, value)

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

        self._initialize_curvature_vectors()

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
        v1 = np.zeros(3, dtype=np.float32)
        v2 = np.zeros(3, dtype=np.float32)
        cdef float[:] v1_view = v1
        cdef float[:] v2_view = v2
        
        Mvi = np.zeros((3,3), dtype=np.float32)
        cdef float[:,:] Mvi_view = Mvi

        areas = np.zeros(self._vertices.shape[0], dtype=np.float32)
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
        
        self._pE = np.exp(-(1.0/KBT)*self._E)  # unitless
        
        ## Take into account the change in neighboring energies for each
        # vertex shift
        # Compute dEdN by component
        dEdN_H = areas*self.kc*(2.0*self._H-self.c0)*self._dH  # eV/nm
        dEdN_K = areas*self.kg*self._dK  # eV/nm
        dEdN_sum = (dEdN_H + dEdN_K + self._dE_neighbors) # eV/nm # + dE_neighbors)
        dEdN = -1.0*dEdN_sum # eV/nm # *(1.0-self._pE)

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

    cdef point_attraction_grad_kdtree(self, np.ndarray points, np.ndarray sigma, float w=0.95, int search_k=200):
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

        dirs = np.zeros((n_verts,3), dtype=np.float32)
        attraction = np.zeros(3, dtype=np.float32)

        charge_sigma = self._mean_edge_length/2.5  # nm
        charge_var = (2*charge_sigma**2)  # nm^2

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
            
            try:
                d = self._vertices['position'][i,:] - points[neighbors]  # nm
            except(IndexError):
                raise IndexError('Could not access neighbors for position {}.'.format(self._vertices['position'][i,:]))
            dd = (d*d).sum(1)  # nm^2

            pt_weight_matrix = 1. - w*np.exp(-dd/charge_var)  # unitless
            pt_weights = np.prod(pt_weight_matrix)  # unitless
            r = np.sqrt(dd)/sigma[neighbors]  # unitless
            
            rf = -(1-r**2)*np.exp(-r**2/2) + (1-np.exp(-(r-1)**2/2))*(r/(r**3 + 1))  # unitless
            # Points at the vertex we're interested in are not de-weighted by the
            # pt_weight_matrix
            rf = rf*(pt_weights/pt_weight_matrix) # unitless
            
            attraction[:] = (-d*(rf/np.sqrt(dd))[:,None]).sum(0)  # unitless
            attraction_norm = np.linalg.norm(attraction)
            attraction[:] = (attraction*np.prod(1-np.exp(-r**2/2)))/attraction_norm  # unitless
            attraction[attraction_norm == 0] = 0  # div by zero
            dirs[i,:] = attraction

        return dirs

    def delaunay_remesh(self, points, eps=1):
        print('Delaunay remesh...')

        # Generate tesselation from mesh control points
        v = self._vertices['position'][self._vertices['halfedge']!=-1]
        d = scipy.spatial.Delaunay(v)
        
        # Ensure all simplex vertices are wound s.t. normals point away from simplex centroid
        tri = delaunay_utils.orient_simps(d, v)

        # Remove simplices outside of our mesh
        ext_inds = delaunay_utils.greedy_ext_simps(tri, self)
        simps = delaunay_utils.del_simps(tri, ext_inds)

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

    #################################
    # Functions to fix mesh topology
    # 
    # In general we must assume that our starting mesh does not always accurately capture
    # the true toplology of the object. As we refine the mesh, we must also be able to correct
    # these topological errors. Topological errors fall into 3 categories:
    #
    # 1) objects which should not be connected but are connected (necks). In general, the 
    #    shinkwrapping algorithm will pull the surface in to a constricted "neck" which
    #    we must sever.
    #
    # 2) objects which are not connected but which should be connected (fusion).
    #
    # 3) objects which should have "holes" / fenestrations but do not. This can also be
    #    considered as a special case of 2) where an object should be connected to itself.
    #
    # We currently have methods to deal with cases 1 and 3 (necks and holes), but lack handling
    # for more generic object fusion. It is possible that this could be obtained by relaxing the
    # connected component and sign testing in the hole punching code.
    #
    # There are several avenues for potential improvement, but the current code is functional

    def _holepunch_punch_hole(self, np.int32_t face0, np.int32_t face1):
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
        self._holepunch_insert_square(self._cfaces[face0].halfedge, 
                            self._cfaces[face1].halfedge,  
                            <np.int32_t *> np.PyArray_DATA(n_edges), 
                            <np.int32_t *> np.PyArray_DATA(n_faces), 
                            n_edge_idx, n_face_idx)

        n_face_idx += 2
        n_edge_idx += 6
        
        self._holepunch_insert_square(self._chalfedges[self._cfaces[face0].halfedge].prev, 
                            self._chalfedges[self._cfaces[face1].halfedge].next,  
                            <np.int32_t *> np.PyArray_DATA(n_edges), 
                            <np.int32_t *> np.PyArray_DATA(n_faces), 
                            n_edge_idx, n_face_idx)

        n_face_idx += 2
        n_edge_idx += 6

        self._holepunch_insert_square(self._chalfedges[self._cfaces[face0].halfedge].next, 
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

    def _holepunch_punch_hole2(self, np.ndarray component_cands, np.ndarray paired_component_cands):
        inner_boundary0 = self._holepunch_component_boundary(component_cands)
        inner_boundary1 = self._holepunch_component_boundary(paired_component_cands)

        # Find the closest position to the first position in boundary 0
        position0 = self._vertices['position'][self._halfedges['vertex'][inner_boundary0[0]]]
        positions1 = self._vertices['position'][self._halfedges['vertex'][inner_boundary1]]
        min_idx = np.argmin((positions1 - position0)**2)
        
        # roll positions 1 back by min_idx+1, so the closest vertex to inner_boundary[0]
        # is inner_boundary1[0].prev's vertex
        inner_boundary1 = np.roll(inner_boundary1, -min_idx-1)

        # Store all vertices used by these faces, for later
        def face_vertices(candidates):
            e0 = self._faces['halfedge'][candidates]
            e1 = self._halfedges['next'][e0]
            e2 = self._halfedges['prev'][e0]
            v0 = self._halfedges['vertex'][e0]
            v1 = self._halfedges['vertex'][e1]
            v2 = self._halfedges['vertex'][e2]

            return np.hstack([v0, v1, v2])
        face_vertices = np.hstack([face_vertices(component_cands),
                                    face_vertices(paired_component_cands)])

        # Convert the inner boundary to the outer boundary, which will be
        # left after face deletion 
        def find_outer_boundary(inner_boundary):
            boundary = np.zeros_like(inner_boundary)
            for j, edge in enumerate(inner_boundary):
                twin = self._chalfedges[edge].twin
                boundary[j] = twin
                # reassign vertex so its halfedge will still exist
                self._cvertices[self._chalfedges[edge].vertex].halfedge = twin
                # Disconnect the halfedges
                self._chalfedges[twin].twin = -1
            return boundary
    
        # Reverse the boundaries so they run in the correct order
        boundary0 = find_outer_boundary(inner_boundary0)
        boundary1 = find_outer_boundary(inner_boundary1)
        boundary_polygons = np.hstack([boundary0, boundary1])

        # Add one square to connect the separated boundaries
        # (TODO: This could be done with a single triangle)
        n_faces, n_edges = self.new_faces(2), self.new_edges(6)
        n_face_idx, n_edge_idx = 0, 0
        self._holepunch_insert_square(inner_boundary0[0], inner_boundary1[0],
                                        <np.int32_t *> np.PyArray_DATA(n_edges), 
                                        <np.int32_t *> np.PyArray_DATA(n_faces), 
                                        n_edge_idx, n_face_idx)

        # Update the boundary to include two new edges
        boundary_polygons[0] = n_edges[2]
        boundary_polygons[len(boundary0)] = n_edges[5]

        # Delete faces
        for face in component_cands:
            self._face_delete(self._cfaces[face].halfedge)
        for face in paired_component_cands:
            self._face_delete(self._cfaces[face].halfedge)

        # Delete any vertices that were in the faces, but aren't in the
        # outer boundaries
        boundary_vertices = self._halfedges['vertex'][boundary_polygons]
        remaining_vertices = set(face_vertices) - set(boundary_vertices)
        for vertex in remaining_vertices:
            self._vertex_delete(vertex)

        # Zipper the boundary
        n_edges = boundary_polygons.shape[0]

        new_faces = self.new_faces(int(n_edges-2))
        new_edges = self.new_edges(int(3*(n_edges-3)+3))

        self._zig_zag_triangulation(np.atleast_2d(boundary_polygons), 
                                    <np.int32_t *> np.PyArray_DATA(new_edges), 
                                    <np.int32_t *> np.PyArray_DATA(new_faces), 
                                    0, n_edges, live_update=True)

        self._clear_flags()
        self.face_normals
        self.vertex_neighbors

    cdef _holepunch_insert_square(self, np.int32_t edge0, np.int32_t edge1, 
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

    def _holepunch_find_candidate_faces(self, points, eps=10.0):
        """
        Find all mesh faces that have no points within a distance eps of their center. 
        Return the index of these faces.
        """
        tree = scipy.spatial.cKDTree(points)
        dist, _ = tree.query(self._vertices['position'][self.faces].mean(1))
        
        inds = np.flatnonzero(self._faces['halfedge'] != -1).astype('i4')

        return inds[dist>eps] # Optionally, (mesh._mean_edge_length + eps)], but this seems to work worse

    cdef _c_holepunch_pair_candidate_faces(self, int[:] candidates, int n_candidates, int[:] pairs):
        c_holepunch_pair_candidate_faces(&(self._cvertices[0]), 
                                         &(self._cfaces[0]),
                                         &(self._chalfedges[0]),
                                         &(candidates[0]),
                                         n_candidates,
                                         &(pairs[0]))

    def _holepunch_pair_candidate_faces(self, np.ndarray candidates):
        """
        For each face, find the opposing face with the nearest centroid that has a
        normal in the opposite direction of this face and form a pair. Note this pair
        does not need to be unique.
        """
        if USE_C:
            pairs = -1*np.ones(candidates.shape[0], dtype='i4')  # index of paired face for each candidate
            self._c_holepunch_pair_candidate_faces(candidates, candidates.shape[0], pairs)
            pair_inds = pairs!=-1  # some faces may have no pairs
            new_inds = np.cumsum(pair_inds)-1

            return candidates[pair_inds], new_inds[pairs[pair_inds]]
            
        else:
            # CAUTION: This has a tendency to blow up memory usage.

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
            shift = candidate_shift - n_hat*(((n_hat*candidate_shift).sum(2)*np.linalg.norm(candidate_shift*candidate_shift, axis=2))[...,None])
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

    def _holepunch_empty_prism_candidate_faces(self, points, candidates, candidate_pair, eps=10.0):
        """
        For each candidate pair, check that there are no points in between the candidate triangles.
        This expects candidate, candidate_pair output from _holepunch_pair_candidate_faces(), where the
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
                kept_cands[i] |= True
                disallowed[candidates == candidates[j]] |= True
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
            empty = np.sum(below_hp0_ci & below_hp1_ci & below_hp2_ci \
                           & below_hp0_cj & below_hp1_cj & below_hp2_cj) == 0
            
            kept_cands[i] |= empty
            disallowed[candidates == candidates[j]] |= empty
            
        c = candidates[kept_cands]
        cp = candidates[candidate_pair[kept_cands]]
        
        return np.hstack([c,cp]), np.hstack([range(len(c),2*len(c)), range(len(c))])

    def _holepunch_connect_candidates(self, candidates):
        """
        Compute the connected component labeling of the kept faces 
        such that faces that share edges are considered connected.
        """
        # mesh._components_valid = 0
        self._faces['component'][:] = 1e6  # -1 out the componenets
        
        # Give each face its own component
        self._faces['component'][candidates] = range(len(candidates))
        
        for _ in range(2):
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

    def _holepunch_component_euler_characteristic(self, candidates, component):
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

    def _holepunch_update_topology(self, candidates, candidate_pairs, component, euler):
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
                    if component[pair_idx] == c:
                        continue
                    pair_component_idx = np.argmax(unique_components==component[pair_idx])
                    if used_components[pair_component_idx]:
                        continue
                    paired_component_cands = candidates[component == component[pair_idx]]

                    # self._holepunch_punch_hole(component_cands[j], candidates[pair_idx])
                    self._holepunch_punch_hole2(component_cands, paired_component_cands)
                    
                    used_components[i] = True
                    used_components[pair_component_idx] = True
                    break
            else:
                print(f"Component {c} has Euler characteristic {euler[i]}. I don't know what to do with this.")
            
            # Mark this component as used
            used_components[i] = True

    def _holepunch_component_boundary(self, candidates):
        """Take a list of face indices and return the halfedges forming the boundary of this component
        in the order of the external boundary (as if we are traversing around the boundary in the
        direction opposite its halfedge flow)."""

        cdef int j
        cdef np.int32_t edge, twin_vertex

        e0 = self._faces['halfedge'][candidates]
        e1 = self._halfedges['next'][e0]
        e2 = self._halfedges['prev'][e0]

        he = np.hstack([e0,e1,e2])

        boundary_edges = list(set(he) - set(self._halfedges['twin'][he]))
        ordered_boundary = np.zeros((len(boundary_edges),), dtype='i4')

        # Now order them by finding pivot vertices
        edge = boundary_edges.pop()
        ordered_boundary[0] = edge
        twin_vertex = self._chalfedges[self._chalfedges[edge].twin].vertex
        j = 1
        failsafe = 100 + ordered_boundary.shape[0]
        while (j < ordered_boundary.shape[0]) and (failsafe > 0):
            for i, e in enumerate(boundary_edges):
                if self._chalfedges[e].vertex == twin_vertex:
                    edge = boundary_edges.pop(i)
                    ordered_boundary[j] = edge
                    twin_vertex = self._chalfedges[self._chalfedges[edge].twin].vertex
                    j += 1
                    break
            failsafe -= 1

        return ordered_boundary

    def punch_holes(self, pts, eps=10.0):
        """
        Create holes in the mesh if there are opposing faces with no points (pts) within eps of 
        the prism formed between them.

        Parameters
        ----------
        pts : np.array
            (N, 3) array of point locations
        eps : float
            Distance to closest point
        """
        # Find all mesh faces that have no points within eps of their face center
        hc = self._holepunch_find_candidate_faces(pts, eps=eps)  # TODO: 5.0 is empirical

        if len(hc) < 1:
            return

        # Pair these faces by matching each face to its closest face in mean normal space
        # with an opposing normal. Allows many-to-one.
        cands, pairs = self._holepunch_pair_candidate_faces(hc)

        # Check if there are no points within eps of the prism formed by each face pair. Keep these
        # only. Restores one-to-one face matching.
        empty_cands, empty_pairs = self._holepunch_empty_prism_candidate_faces(pts, cands, pairs, eps=eps)

        if len(empty_cands) < 1:
            return

        # Group the remaining faces by edge connectivity.
        component = self._holepunch_connect_candidates(empty_cands)

        # Compute the euler characteristic of each component. Euler 0 = tube, 1 = plane patch.
        chi = self._holepunch_component_euler_characteristic(empty_cands, component)

        # Punch holes between place patches (cut tubes is currently disabled)
        self._holepunch_update_topology(empty_cands, empty_pairs, component, chi)

    def remove_necks(self, neck_curvature_threshold_low=-1e-4, neck_curvature_threshold_high=1e-2):
        """
        Remove necks, using high -ve Gaussian curvature as a marker for candidate necks.

        When compared to holes, necks are an easier prospect as one can simple delete all vertices/faces
        below threshold and repair.

        TODO: Improve neck selection by looking at, e.g. point_influence as well. 
        """

        verts = np.flatnonzero((self.curvature_gaussian < neck_curvature_threshold_low)|(self.curvature_gaussian > neck_curvature_threshold_high))
        self.unsafe_remove_vertices(verts)
        self.repair()
        #self.repair()
        self.remesh()

    # End topology functions
    ##########################
    
    #cdef grad(self, np.ndarray points, np.ndarray sigma):
    #    """
    #    Gradient between points and the surface.
    #
    #    Parameters
    #    ----------
    #        points : np.array
    #            3D point cloud to fit.
    #        sigma : float
    #            Localization uncertainty of points.
    #    """
    #
    #    dN = 0.1
    #    if USE_C:
    #        curvature = self.curvature_grad_c(dN=dN, skip_prob=self.skip_prob)
    #    else:
    #        curvature = self.curvature_grad(dN=dN, skip_prob=self.skip_prob)
    #    attraction = self.point_attraction_grad_kdtree(points, sigma, w=0.95, search_k=self.search_k)
    #
    #    print("Curvature: {}".format(np.mean(curvature,axis=0)))
    #    print("Attraction: {}".format(np.mean(attraction,axis=0)))
    #    print("Curvature-to-attraction: {}".format(np.mean(curvature/attraction,axis=0)))
    #
    #    g = self.a*attraction + self.c*curvature
    #    print("Gradient: {}".format(np.mean(g,axis=0)))
    #    return g

    #def opt_adam(self, points, sigma, max_iter=250, step_size=1, beta_1=0.9, beta_2=0.999, eps=1e-8, **kwargs):
    #    """
    #    Performs Adam optimization (https://arxiv.org/abs/1412.6980) on
    #    fit of surface mesh surf to point cloud points.
    #
    #    Parameters
    #    ----------
    #        points : np.array
    #            3D point cloud to fit.
    #        sigma : float
    #            Localization uncertainty of points.
    #    """
    #    # Initialize moment vectors
    #    m = np.zeros(self._vertices['position'].shape)
    #    v = np.zeros(self._vertices['position'].shape)
    #
    #    t = 0
    #    # g_mag_prev = 0
    #    # g_mag = 0
    #    while (t < max_iter):
    #        print('Iteration %d ...' % t)
    #         
    #        t += 1
    #        # Gaussian noise std
    #        noise_sigma = np.sqrt(self.step_size / ((1 + t)**0.55))
    #        # Gaussian noise
    #        noise = np.random.normal(0, noise_sigma, self._vertices['position'].shape)
    #        # Calculate graident for each point on the  surface, 
    #        g = self.grad(points, sigma)
    #        # add Gaussian noise to the gradient
    #        g += noise
    #        # Update first biased moment 
    #        m = beta_1 * m + (1. - beta_1) * g
    #        # Update second biased moment
    #        v = beta_2 * v + (1. - beta_2) * np.multiply(g, g)
    #        # Remove biases on moments & calculate update weight
    #        a = step_size * np.sqrt(1. - beta_2**t) / (1. - beta_1**t)
    #        # Update the surface
    #        self._vertices['position'] += a * m / (np.sqrt(v) + eps)

    #def opt_euler(self, points, sigma, max_iter=100, step_size=1, eps=0.00001, **kwargs):
    #    """
    #    Normal gradient descent.
    #
    #    Parameters
    #    ----------
    #        points : np.array
    #            3D point cloud to fit.
    #        sigma : float
    #            Localization uncertainty of points.
    #    """
    #
    #    # Precalc 
    #    dr = (self.delaunay_remesh_frequency != 0)
    #    r = (self.remesh_frequency != 0)
    #    if r:
    #        initial_length = self._mean_edge_length
    #        final_length = 3*np.max(sigma)
    #        m = (final_length - initial_length)/max_iter
    #   
    #   for _i in np.arange(max_iter):
    #
    #        print('Iteration %d ...' % _i)
    #        
    #        # Calculate the weighted gradient
    #        shift = step_size*self.grad(points, sigma)
    #
    #        # Update the vertices
    #        self._vertices['position'] += shift
    #
    #        # self._faces['normal'][:] = -1
    #        # self._vertices['neighbors'][:] = -1
    #        self._face_normals_valid = 0
    #        self._vertex_normals_valid = 0
    #        self.face_normals
    #        self.vertex_neighbors
    #
    #        # If we've reached precision, terminate
    #        if np.all(shift < eps):
    #           break
    #
    #        if (_i == 0):
    #            # Don't remesh
    #            continue
    #
    #        # Remesh
    #        if r and ((_i % self.remesh_frequency) == 0):
    #            target_length = initial_length + m*_i
    #            self.remesh(5, target_length, 0.5, 10)
    #            print('Target mean length: {}   Resulting mean length: {}'.format(str(target_length), 
    #                                                                            str(self._mean_edge_length)))
    #
    #        # Delaunay remesh
    #        if dr and ((_i % self.delaunay_remesh_frequency) == 0):
    #            self.delaunay_remesh(points, self.delaunay_eps)

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
            if kwargs.get('minimum_edge_length', -1) < 0:
                final_length = np.clip(np.min(sigma)/2.5, 1.0, 50.0)
            else:
                final_length = kwargs.get('minimum_edge_length')
            
            # We want face area, rather than edge length to decrease linearly as iterations proceed
            # this means we should be linear in edge length squared
            initial_length_2 = initial_length*initial_length
            final_length_2 = final_length*final_length

            m = (final_length_2 - initial_length_2)/max_iter

        neck_first_iter = getattr(self, 'neck_first_iter', -1)


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
                # self.delaunay_remesh(points, self.delaunay_eps)
                self.punch_holes(points, self.delaunay_eps)
                #self.remove_necks()
                # break

            # Remesh
            if r and ((j % self.remesh_frequency) == 0):
                if (neck_first_iter > 0) and (j > neck_first_iter):
                    self.remove_necks(getattr(self, 'neck_threhold_low', -1e-4), getattr(self, 'neck_threshold_high', 1e-2)) # TODO - do this every remesh iteration or not?

                target_length = np.sqrt(initial_length_2 + m*(j+1))
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

    @property
    def point_influence(self):
        """
        An attempt to measure how constrained a given vertex is by the points

        (Essentially just Ahfunc(I))
        """

        s = self.cg.Ahfunc(np.ones_like(self.cg.res)).reshape(self.vertices.shape)
        return np.sqrt((s*s).sum(1))

    #@property
    #def _S1(self):
    #    """ Search direction to minimise regularisation term (curvature)""""
    #    raise NotImplementedError('this should mirror S1 in subseearch')
        
    def shrink_wrap(self, points=None, sigma=None, method='conjugate_gradient', max_iter=None, **kwargs):

        if method not in DESCENT_METHODS:
            print('Unknown gradient descent method. Using {}.'.format(DEFAULT_DESCENT_METHOD))
            method = DEFAULT_DESCENT_METHOD

        if max_iter is None:
            max_iter = self.max_iter
        
        # save points and sigmas so we can call again to continue
        if points is None:
            points = self._points

        if sigma is None:
            sigma = self._sigma
        
        opts = dict(points=points,
                    sigma=sigma, 
                    max_iter=max_iter,
                    step_size=self.step_size, 
                    beta_1=self.beta_1, 
                    beta_2=self.beta_2,
                    eps=self.eps,
                    **kwargs)

        self._points = points
        self._sigma = sigma

        return getattr(self, 'opt_{}'.format(method))(**opts)

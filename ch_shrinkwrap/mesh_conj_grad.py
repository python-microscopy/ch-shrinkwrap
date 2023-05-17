#!/usr/bin/python

###################################
# Based on PYME.Deconv.dec dec.py
###################################

from numpy.compat.py3k import npy_load_module
from .delaunay_utils import voronoi_poles

import numpy as np
import scipy.spatial

USE_C = True

if USE_C:
    from . import conj_grad_utils

from .conj_grad import TikhonovConjugateGradient

class ShrinkwrapMeshConjGrad(TikhonovConjugateGradient):
    _search_k = 200
    _points = None
    _vertices = None
    _vertex_neighbors = None
    _faces = None
    _face_neighbors = None
    _sigma = None
    _search_rad = 100
    N, M = None, None
    dims, shape = None, None
    d = None

    def __init__(self, mesh, points, sigma=None, search_k=200, search_rad=100, shield_sigma=None, use_octree=False):
        TikhonovConjugateGradient.__init__(self)
        
        #self.Lfuncs, self.Lhfuncs = ["Lfunc3", "I"], ["Lfunc3", "I"]
        #self.Lfuncs, self.Lhfuncs = ["Lfunc3"], ['Lhfunc3']
        self.Lfuncs, self.Lhfuncs = ["I"], ["I"]
        #self.Lfuncs, self.Lhfuncs = ["wfunc"], ["wfunc"]
        self.mesh = mesh
        self.points = points
        self.sigma = sigma
        
        self._mesh_vertex_mask = mesh._vertices['halfedge'] != -1
        
        self.vertices = mesh._vertices['position'] #[self._mesh_vertex_mask]
        self.faces = mesh.faces
        
        # cache vertex neighbours
        n = mesh._halfedges['vertex'][mesh._vertices['neighbors']] # [self._mesh_vertex_mask]]
        n_idxs = mesh._vertices['neighbors'] == -1 #[self._mesh_vertex_mask] == -1
        n[n_idxs] = -1

        self.vertex_neighbors = n


        
        #self.vertices, self.vertex_neighbors = vertices, vertex_neighbors
        #self.faces, self.face_neighbors = faces, face_neighbors
        
        self.search_k = search_k
        self.search_rad = search_rad
        self._prev_loopcount = -1

        self._use_octreee = use_octree #toggle between octree and kDTree for calculating weight matrix


    
    def prep(self):
        pass
        
    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices):
        self._vertices = vertices
        self.M = vertices.shape[0]
        self.dims = vertices.shape[1]
        self.shape = vertices.shape  # hack
        
    @property
    def vertex_neighbors(self):
        return self._vertex_neighbors
    
    @vertex_neighbors.setter
    def vertex_neighbors(self, neighbors):
        self._vertex_neighbors = neighbors
        if self.N:
            assert(self._vertex_neighbors.shape[1] == self.N)
        else:
            self.N = self._vertex_neighbors.shape[1]

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        self._faces = faces

    @property
    def face_neighbors(self):
        return self._face_neighbors
    
    @face_neighbors.setter
    def face_neighbors(self, neighbors):
        self._face_neighbors = neighbors
        if self.N:
            assert(self._face_neighbors.shape[1] == self.N)
        else:
            self.N = self._face_neighbors.shape[1]
    
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma
    
    @property
    def points(self):
        return self._points
    
    @points.setter
    def points(self, pts):
        self._points = pts
        self._tree = scipy.spatial.cKDTree(self._points)
        
    @property
    def search_k(self):
        return self._search_k

    @search_k.setter
    def search_k(self, search_k):
        self._search_k = min(search_k, self.points.shape[0])

    @property
    def search_rad(self):
        return self._search_rad

    @search_rad.setter
    def search_rad(self, search_rad):
        self._search_rad = max(search_rad, 1.0)

    
        
    def search(self, data, lams, defaults=None, num_iters=10, weights=None, sigma_inv=1.0, pos=False, last_step=True):
        """Custom search to add weighting to res
        """

        self._prev_loopcount = -1

        if weights is None:
            # allow sigma and weights to be different (for image fitting)
            weights = sigma_inv

        if not np.isscalar(weights):
            self.mask = weights > 0
            weights = weights / weights.mean()
        else:
            self.mask = np.isfinite(data.ravel())

        #self._sigma = sigma

        # guess a starting estimate for the object
        # NOTE: start_guess must return a unique object (e.g. a copy() of data)
        self.fs = self.start_guess(data)

        # create a flattened view of our result
        self.f = self.fs.ravel()

        # Assume defaults are 0 for each regularization term if we don't pass any explicitly
        if defaults is None:
            defaults = [self.default_guess(0) for x in self.Lfuncs]

        #make things 1 dimensional
        data = data.ravel()
        self.res = 0*data

        #number of search directions
        n_smooth = min(len(self.Lfuncs), len(lams))
        n_search = n_smooth+1  # inital number of search directions is search along Afunc + Lfuncs
        s_size = n_search+1    # eventually we'll search along fnew - self.f

        # construct pairs to compare for search metric
        a = np.arange(n_search)
        search_pairs = []
        for i in a:
            for j in a[1:]:
                if i == j:
                    continue
                search_pairs.append((i,j))
        n_pairs = len(search_pairs)

        # listify lams if only looking along 2 directions
        # (one regularization term)
        if type(lams) is float:
            lams = [lams]

        # directions for regularized parameters
        prefs = np.zeros((np.size(self.f), n_smooth), 'f')

        #initial search directions
        S = np.zeros((np.size(self.f), s_size), 'f')

        # replace any empty lambdas with zeros
        #if len(lams) < len(self.Lfuncs):
        #    print(f"not enough lambdas, defaulting {len(self.Lfuncs)-len(lams)} of them to 0")
        #    tmp = lams
        #    lams = [0]*len(self.Lfuncs)
        #    lams[:len(tmp)] = tmp

        self.loopcount = 0

        while (self.loopcount  < num_iters) and (not self._stop_cond()):
            self.loopcount += 1
            
            # residuals
            self.res[:] = (weights*(data - self.Afunc(self.f)))

            defaults = [self._defaults(i) for i in range(n_smooth)]

            # weight the residuals based on distance
            # /8 = 2 * 2^2 for weighting within 2*sigma of the point
            # w = np.exp(-(self.d.ravel()**2)*((weights/2)**2)) + 1/(self.d.ravel()**2+1)
            #w = 0.5-np.arctan(self.d.ravel()**2-2.0/weights**2)/np.pi

            w = 1.0/(self.d.ravel()*sigma_inv/2.0+1)
            
            #md = np.median(self.d)
            #print(f'md:{md}')
            #w = md/(self.d.ravel() + md)
            
            #w = w*w

            #w = 2.0/(self.d.ravel()*.01 + 1)
            
            #w = 1.0
            #w = 1.0/(self.d.ravel()/2.0+1)
            #w = weights
            # w = np.exp(-(self.d.ravel()*weights/2)**2)
            # w = 1.0/np.log((2*self.d.ravel()/weights)**2+np.exp(1))
            # print("WEIGHTING")
            # print(w)
            self.res *= w

            #print ('res:', self.res.reshape(-1, self.dims))

            # search directions
            S[:,0] = self.Ahfunc(self.res)
            #print(S.shape, prefs.shape, defaults.shape)
            for i in range(n_smooth):
                #print(i)
                prefs[:,i] = getattr(self, self.Lfuncs[i])(self.f - defaults[i]) # residuals
                S[:,i+1] = -1.0*getattr(self, self.Lhfuncs[i])(prefs[:,i])
            
            # check to see if the search directions are orthogonal
            # this can be used as a measure of convergence and a stopping criteria
            test = 1.0
            for pair in search_pairs:
                test -= ((1.0/n_pairs) * abs((S[:,pair[0]]*S[:,pair[1]]).sum()
                        / (np.linalg.norm(S[:,pair[0]])*np.linalg.norm(S[:,pair[1]]))))

            #print & log some statistics
            #print(('Test Statistic %f' % (test,)))
            self.tests.append(test)
            self.ress.append(np.linalg.norm(self.res))
            self.prefs.append(np.linalg.norm(prefs,axis=0))

            #minimise along search directions to find new estimate
            fnew, self.cpred, self.wpreds = self.subsearch(self.f, self.res[self.mask], defaults, self.Afunc, self.Lfuncs, lams, S[:, 0:n_search])

            #positivity constraint (not part of original algorithm & could be ommitted)
            if pos:
                fnew = (fnew*(fnew > 0))

            #add last step to search directions, as per classical conj. gradient
            if last_step:
                S[:,(s_size-1)] = (fnew - self.f)
                n_search = s_size

            self.S = S

            #set the current estimate to out new estimate
            self.f[:] = fnew
            self.mesh._vertices['position'][self._mesh_vertex_mask] = fnew.reshape(self.vertices.shape)[self._mesh_vertex_mask]
            self.mesh._initialize_curvature_vectors()

        return np.real(self.fs)
        
    def _compute_weight_matrix(self, f, w=0.95, shield_sigma=10):
        """
        Each point pulls on its nearest face.

        find which vertices are tied to which points.

        To start with, make each point act on it's nearest face (TODO - nearest N face and smoothing?)
        computes a sparse representation of the influence matrix: 
        - an n_points x 3 array of vertex indices corresponding to the nearest face (which vertices each point operates on)
        - an n_points x 3 array of weights  (1 - d_j/sum(d_j)) where d_j is the distance from the point to the each of the 3 vertices

        """
        #dd = np.zeros((self.M,self.points.shape[0]), dtype='f')

        if False: #USE_C:
            # we don't have a c version of this yet
            conj_grad_utils.c_compute_weight_matrix(np.ascontiguousarray(f), self.vertex_neighbors, self.points, dd, self.dims, self.points.shape[0], self.M, self.N, shield_sigma, self.search_rad)
        else:
            fv = f.reshape(-1,self.dims)
            
            import scipy.spatial

            # Create a list of face centroids for search
            face_centers = fv[self._faces].mean(1)

            # Construct a kdtree over the face centers
            tree = scipy.spatial.cKDTree(face_centers)

            # Get k closet face centroids for each point
            _, _faces = tree.query(self.points, k=1)
            
            #print(self._faces.shape, _faces.shape)
            
            # vertex indices (n_points x 3)
            v_idx = self._faces[_faces, :]

            #compute distances
            d = np.zeros(v_idx.shape, 'f4')
            
            for j in range(3):
                d_ij = (fv[v_idx[:,j]] - self.points) # vector distance
                d[:, j] = np.sqrt(np.sum(d_ij *d_ij, 1)) # scalar distance

            #print(self.points.shape, v_idx.shape, d.shape, d_ij.shape)
            
            w = 1.0/np.maximum(d, 1e-6)

            w = w/w.sum(1)[:,None]

            #print(d, d/d.sum(1)[:,None], w)
            assert(not np.any(np.isnan(w)))
            
            return v_idx, w 
    
    def _compute_weight_matrix2(self, f, w=0.95, shield_sigma=20):
        """Each face pulls toward its nearest point"""
        fv = f.reshape(-1,self.dims)
            
        import scipy.spatial

        # Create a list of face centroids for search
        face_centers = fv[self._faces].mean(1)

        # Find the closest point to each face
        _, _points = self._tree.query(face_centers, k=1)
        
        #print(self._faces.shape, _faces.shape)

        v_idx = np.zeros(self.points.shape, 'i4')
        v_idx[_points,:] = self._faces[:,:]

        #compute distances
        d = np.zeros(v_idx.shape, 'f4')
        
        for j in range(3):
            d_ij = (fv[v_idx[:,j]] - self.points) # vector distance
            d[:, j] = np.sqrt(np.sum(d_ij *d_ij, 1)) # scalar distance
        d[v_idx == 0] = 0

        #print(self.points.shape, v_idx.shape, d.shape, d_ij.shape)
        
        w = 1.0/np.maximum(d, 1e-6)

        w = w/w.sum(1)[:,None]
        w[v_idx[:,0] == 0,:] = 0

        #print(d, d/d.sum(1)[:,None], w)
        assert(not np.any(np.isnan(w)))
        
        return v_idx, w 

    def _compute_weight_matrix3(self, f, w=0.95, shield_sigma=20):
        """
        For any face experiencing no pull, its nearest neighbor point pulls on it.
        """

        fv = f.reshape(-1,self.dims)
        
        import scipy.spatial

        # Create a list of face centroids for search
        face_centers = fv[self._faces].mean(1)

        # Construct a kdtree over the face centers
        tree = scipy.spatial.cKDTree(face_centers)

        # Get k closet face centroids for each point
        _, _faces = tree.query(self.points, k=1)

         # Find faces that experienced no pull
        _unassigned_faces = np.array(list(set(range(self._faces.shape[0]))-set(_faces))).astype('i4')
        # print(_unassigned_faces)

        # Find the closest point to each unassinged face
        _, _points = self._tree.query(face_centers[_unassigned_faces,:], k=1)
                
        # vertex indices (n_points x 3)
        v_idx = np.zeros(self.points.shape, 'i4')
        v_idx[_points,:] = self._faces[_unassigned_faces,:]

        #compute distances
        d = np.zeros(v_idx.shape, 'f4')
        for j in range(3):
            d_ij = (fv[v_idx[:,j]] - self.points) # vector distance
            d[:, j] = np.sqrt(np.sum(d_ij *d_ij, 1)) # scalar distance
        d[v_idx == 0] = 0

        #print(self.points.shape, v_idx.shape, d.shape, d_ij.shape)
        
        w = 1.0/np.maximum(d, 1e-6)

        w = w/w.sum(1)[:,None]
        w[v_idx[:,0] == 0,:] = 0

        #print(d, d/d.sum(1)[:,None], w)
        assert(not np.any(np.isnan(w)))
        
        return v_idx, w 

    def _compute_weight_matrix4(self, f, w=0.95, shield_sigma=20):
        """
        Each point pulls on its nearest face, but also weighted by distance.
        """
        fv = f.reshape(-1,self.dims)
        
        import scipy.spatial
        from PYME.experimental import octree

        # Create a list of face centroids for search
        face_centers = fv[self._faces].mean(1)

        

        if not self._use_octreee:
            ####
            # KDTREE
            # Construct a kdtree over the face centers
            tree = scipy.spatial.cKDTree(face_centers)

            # Get k closet face centroids for each point
            dmean, _faces = tree.query(self.points, k=1, workers=-1)
            

            # END KDTREE
            ############
        else:
            ########
            # OCTREE
            #
            # NOTE: this is faster (~10x), but currently inexact - it finds the node where the point would be placed in the octree
            # rather than strictly the nearest neighbour - the true nearest neighbour might be in one of the neighbouring cells.
            # Upper bound on error is roughly the size of the octree cell (which will be approx the mesh edge length). 

            tree = octree.gen_octree_from_points({'x' : face_centers[:,0], 'y' : face_centers[:,1], 'z' : face_centers[:,2]}, min_pixel_size=1)

            #_faces_1 = np.array([tree.search(float(p[0]), float(p[1]), float(p[2]))[0] for p in self.points])

            _faces = tree.search_pts(self.points)

            vd = self.points - face_centers[_faces, :]
            dmean = np.sqrt((vd*vd).sum(1))

            #print('vd.shape', vd.shape)
            #print(dmean, dmean_1, _faces, _faces_1)
            #print(np.median(dmean))
            
            # END OCTREE
            ############

        self.d = np.vstack([dmean, dmean, dmean]).T

        #print(self._faces.shape, _faces.shape)
        
        # vertex indices (n_points x 3)
        v_idx = self._faces[_faces, :]

        #compute distances
        d = np.zeros(v_idx.shape, 'f4')
        
        for j in range(3):
            d_ij = (fv[v_idx[:,j]] - self.points) # vector distance
            d[:, j] = np.sqrt(np.sum(d_ij *d_ij, 1)) # scalar distance

        #print(self.points.shape, v_idx.shape, d.shape, d_ij.shape)

        
        # inverse distance weight to ensure that
        # NB - this is for computing face centroid, not 
        # determining point-distance weightings (that wieghting is in search())
        w = 1.0/np.maximum(d, 1e-6) 
        

        #w = np.ones_like(d) 

        #w = np.exp(-((d+1)/100.))

        w = w/w.sum(1)[:,None]
        #w = w/w.sum(1).mean()

        #print(d, d/d.sum(1)[:,None], w)
        assert(not np.any(np.isnan(w)))
        
        return v_idx, w 
    
    def Afunc(self, f):
        """
        Create a map of which (weighted average) vertex each point should drive toward.
        
        f is a set of vertices
        d is the weighted sum of all of the vertices indicating the closest vertex to each point
        """
        if self.calc_w():
            # self.w = self._compute_weight_matrix(self.f)
            # self.w2 = self._compute_weight_matrix3(self.f)
            # self.w = self._compute_weight_matrix2(self.f)
            self.w = self._compute_weight_matrix4(self.f)

            #print(self.w)

        if False: #USE_C:
            #print(self.dims, self.points.shape[0], self.M, self.N)
            conj_grad_utils.c_shrinkwrap_a_func(np.ascontiguousarray(f), self.vertex_neighbors, self.w, d, self.dims, self.points.shape[0], self.M, self.N)
        else:
            fv = f.reshape(-1,self.dims)

            surface_points = np.zeros_like(self.points)

            v_idx, w = self.w
            # v_idx2, w2 = self.w2

            for i in range(3):
                surface_points += fv[v_idx[:,i]]*w[:,i][:,None]
                # surface_points += fv[v_idx2[:,i]]*w2[:,i][:,None]

            assert(not np.any(np.isnan(surface_points)))
            
            #print('a:', surface_points)
            return surface_points.ravel()
    
    def Ahfunc(self, f):
        """
        Map each distance between a point and its closest (weighted average) vertex to
        a new vertex position.
        
        f is a set of points
        d is the weighted sum of all of the points indicating the closest point to each vertex
        """
        
        d = np.zeros([self.M,self.dims], dtype='f')
        fv = f.reshape(-1,self.dims)
        
        if False:#USE_C:
            conj_grad_utils.c_shrinkwrap_ah_func(np.ascontiguousarray(f), self.vertex_neighbors, self.w, d, self.dims, self.points.shape[0], self.M, self.N)
        else:
            v_idx, w = self.w
            # v_idx2, w2 = self.w2
            conj_grad_utils.c_shrinkwrap_ah_helper(v_idx, w, fv.astype('f'), d)
            
            # for j in range(v_idx.shape[0]):
            #     for i in range(3):
            #         d[v_idx[j,i], :] += (w[j,i])*fv[j, :] 
            #     # d[v_idx2[:,i], :] += (w2[:,i][:,None])*fv 

        
        #print(fv.dtype, v_idx.dtype, w.dtype)
        #print('ah:',d)
        assert(not np.any(np.isnan(d)))
        #print('ah:',d)

        # smooth point force across the mesh
        #d[:, 0] = self.mesh.smooth_per_vertex_data(d[:,0])
        #d[:, 1] = self.mesh.smooth_per_vertex_data(d[:,1])
        #d[:, 2] = self.mesh.smooth_per_vertex_data(d[:,2])

        return d.ravel()

    def Lfunc(self, f):
        """
        Minimize distance between a vertex and the centroid of its neighbors.
        """
        # note that f is raveled, by default in C order so 
        # f = [v0x, v0y, v0z, v1x, v1y, v1z, ...] where ij is vertex i, dimension j
        d = np.zeros_like(f)
        if USE_C:
            w = d
            conj_grad_utils.c_shrinkwrap_l_func(np.ascontiguousarray(f), self.vertex_neighbors, w, d, self.dims, self.points.shape[0], self.M, self.N)
        else:
            for i in range(self.M):
                if self.vertex_neighbors[i,0] == -1:
                    # cheat to find self._vertices['halfedge'] == -1
                    continue
                for j in range(self.dims):
                    nn = self.vertex_neighbors[i,:]
                    S = (nn!=-1).sum()
                    for n in nn:
                        if n == -1:
                            break
                        d[i*self.dims+j] += (f[n*self.dims+j] - f[i*self.dims+j])/S
        
        assert(not np.any(np.isnan(d)))
        return d
    
    def Lhfunc(self, f):
        # Now we are transposed, so we want to add the neighbors to d in column order
        # should be symmetric, unless we change the weighting
        d = np.zeros_like(f)
        if USE_C:
            w = d
            conj_grad_utils.c_shrinkwrap_lh_func(np.ascontiguousarray(f), self.vertex_neighbors, w, d, self.dims, self.points.shape[0], self.M, self.N)
        else:
            for i in range(self.M):
                if self.vertex_neighbors[i,0] == -1:
                    # cheat to find self._vertices['halfedge'] == -1
                    continue
                for j in range(self.dims):
                    nn = self.vertex_neighbors[i,:]
                    S = (nn!=-1).sum()
                    for n in nn:
                        if n == -1:
                            break
                        d[n*self.dims+j] += (f[i*self.dims+j] - f[n*self.dims+j])/S
        
        assert(not np.any(np.isnan(d)))
        
        return d

    def Lfunc2(self, f):
        """
        Minimize distance between a vertex and the centroid of its neighbors.
        """
        # note that f is raveled, by default in C order so 
        # f = [v0x, v0y, v0z, v1x, v1y, v1z, ...] where ij is vertex i, dimension j
        d1 = np.zeros_like(f)
        w = d1
        conj_grad_utils.c_shrinkwrap_l_func(np.ascontiguousarray(f), self.vertex_neighbors, w, d1, self.dims, self.points.shape[0], self.M, self.N)
        
        d = np.zeros_like(d1)
        conj_grad_utils.c_shrinkwrap_l_func(d1, self.vertex_neighbors, w, d, self.dims, self.points.shape[0], self.M, self.N)

        d = (d - d1)#*2 + d1
        
        assert(not np.any(np.isnan(d)))
        return d

    def Lhfunc2(self, f):
        """
        Minimize distance between a vertex and the centroid of its neighbors.
        """
        # note that f is raveled, by default in C order so 
        # f = [v0x, v0y, v0z, v1x, v1y, v1z, ...] where ij is vertex i, dimension j
        d1 = np.zeros_like(f)
        w = d1
        conj_grad_utils.c_shrinkwrap_lh_func(np.ascontiguousarray(f), self.vertex_neighbors, w, d1, self.dims, self.points.shape[0], self.M, self.N)
        
        d = np.zeros_like(d1)
        conj_grad_utils.c_shrinkwrap_lh_func(d1, self.vertex_neighbors, w, d, self.dims, self.points.shape[0], self.M, self.N)

        d = (d - d1)#*2 + d1
        
        assert(not np.any(np.isnan(d)))
        return d

    def Lfunc3(self, f):
        d = np.zeros(f.shape, dtype='f4')
        conj_grad_utils.c_shrinkwrap_lw_func(np.ascontiguousarray(f), self.vertex_neighbors, self.f, d, self.dims, self.points.shape[0], self.M, self.N)
        
        assert(not np.any(np.isnan(d)))
        return d

    def Lhfunc3(self, f):
        #d = np.zeros_like(f)
        d = np.zeros(f.shape, dtype='f4')
        conj_grad_utils.c_shrinkwrap_lhw_func(np.ascontiguousarray(f), self.vertex_neighbors, self.f, d, self.dims, self.points.shape[0], self.M, self.N)
        
        assert(not np.any(np.isnan(d)))
        return d

    def Lfunc4(self, f):
        """
        Minimize distance between a vertex and the centroid of its neighbors.
        """
        # note that f is raveled, by default in C order so 
        # f = [v0x, v0y, v0z, v1x, v1y, v1z, ...] where ij is vertex i, dimension j
        d1 = np.zeros_like(f)
        w = self.f
        conj_grad_utils.c_shrinkwrap_lw_func(np.ascontiguousarray(f), self.vertex_neighbors, w, d1, self.dims, self.points.shape[0], self.M, self.N)
        
        d = np.zeros_like(d1)
        conj_grad_utils.c_shrinkwrap_lw_func(d1, self.vertex_neighbors, w, d, self.dims, self.points.shape[0], self.M, self.N)

        d = (d - d1)#*2 + d1
        
        assert(not np.any(np.isnan(d)))
        return d

    def Lhfunc4(self, f):
        """
        Minimize distance between a vertex and the centroid of its neighbors.
        """
        # note that f is raveled, by default in C order so 
        # f = [v0x, v0y, v0z, v1x, v1y, v1z, ...] where ij is vertex i, dimension j
        d1 = np.zeros_like(f)
        w = self.f
        conj_grad_utils.c_shrinkwrap_lhw_func(np.ascontiguousarray(f), self.vertex_neighbors, w, d1, self.dims, self.points.shape[0], self.M, self.N)
        
        d = np.zeros_like(d1)
        conj_grad_utils.c_shrinkwrap_lhw_func(d1, self.vertex_neighbors, w, d, self.dims, self.points.shape[0], self.M, self.N)

        d = (d - d1)#*2 + d1
        
        assert(not np.any(np.isnan(d)))
        return d

    def wfunc(self, f):
        """
        Area-weighting to be used with centroid prior 
        (as mathematically equivalent to lfunc3)

        """

        w = np.zeros(f.shape, 'f4')
        conj_grad_utils.vertex_area_weights(np.ascontiguousarray(self.f), self.vertex_neighbors, w, self.M, self.N)
        return f*w


    # def unconstrained_penalty(self, f):
    #     """
    #     Penalises unconstrained points, making them want to move in the direction of their normal (pull in)
    #     """
    #     fn = f.reshape(self.shape)
    #     n = self.calculate_normals(self.f).reshape(self.shape)

    #     #print('fn, self.pts',fn.shape, self.points.shape)

    #     w = self.Ahfunc(np.ones_like(self.points)).reshape(self.shape)
    #     #print(w.shape)
    #     w = np.sqrt((w*w).sum(1))

    #     #print(w.shape, n.shape)

    #     p = np.maximum(1 - w, 0)[:,None]*n

    #     #print(p.shape)

    #     return p.ravel()

    def _neighbour_centroids(self):
        vnn = self.mesh._halfedges['vertex'][self.mesh.vertex_neighbors]
        mask = self.mesh.vertex_neighbors > -1
        ms= mask.sum(1)
        vc = (self.mesh.vertices[vnn,:]*mask[:,:,None]).sum(1)/ms[:,None]

        vc[ms==0,:] = self.mesh.vertices[ms==0,:]

        return vc  

    def _ncc(self):
        """
        Define a prior for Lfunc to operate against as a location part way
        between the centroid (minimises curvature at the given point) and 
        a point which minimises the curvature at the neighbours. 
        """ 

        vnn = self.mesh._halfedges['vertex'][self.mesh.vertex_neighbors]
        mask = self.mesh.vertex_neighbors > -1
        ms= mask.sum(1)
        
        #centroid
        vc = (self.mesh.vertices[vnn,:]*mask[:,:,None]).sum(1)/ms[:,None]

        #vector from centroid to each of the neighouring vertices
        c_n = self.mesh.vertices[vnn,:] - vc[:,None,:]

        #neighbour normals
        n_n = self.mesh.vertex_normals[vnn, :]

        # alpha = dot(c_n, n_n)/dot(n_n, mesh.vertex_normals)
        # NB - we clip the dot product of the vertex and neighbour normals so as to
        # avoid a div/0 situation as the angle approaches 90 degrees.
        # now clip at 0.5 to avoid being expansive when angles are greater than 60 degrees.
        #alpha = (c_n*n_n).sum(2)/np.maximum((n_n*self.mesh.vertex_normals[:,None,:]).sum(2), 0.5)
        
        n_dot_n = (n_n*self.mesh.vertex_normals[:,None,:]).sum(2)
        alpha = ((c_n*n_n).sum(2))/np.sqrt(2*(np.maximum(n_dot_n, 0) +1))
        
        #print(alpha.shape)
        alpha = (alpha*mask).sum(1)/ms
        
        ### Switch (linearly) between shrinking curvature force and non-shrinking curvature
        # force depending on point influence (how much point-attraction force is acting
        # on the vertex).
        # Low point attraction -> use a shrinking force
        # average/high point attraction -> use a non-shrinking force
        pi = self.mesh.point_influence
        pi = pi #/(pi.sum()/(pi > 0).sum()) # normalise by mean of non-zero entries
        #pi = np.repeat(pi, 3)
        #pi = pi/3.

        #pi = 1

        alpha = alpha*np.minimum(pi**2, 1)
        
        vc = vc + alpha[:,None]*self.mesh.vertex_normals

        vc[ms==0,:] = self.mesh.vertices[ms==0,:]

        return vc

    def _ncc2(self):
        """
        Define a prior for Lfunc to operate against as a location part way
        between the centroid (minimises curvature at the given point) and 
        a point which minimises the curvature at the neighbours. 
        """ 

        vnn = self.mesh._halfedges['vertex'][self.mesh.vertex_neighbors]
        mask = self.mesh.vertex_neighbors > -1
        ms= mask.sum(1)
        
        #centroid
        vc = (self.mesh.vertices[vnn,:]*mask[:,:,None]).sum(1)/ms[:,None]

        #vector from centroid to each of the neighouring vertices
        c_n = self.mesh.vertices[vnn,:] - vc[:,None,:]

        #neighbour normals
        n_n = self.mesh.vertex_normals[vnn, :]

        # alpha = dot(c_n, n_n)/dot(n_n, mesh.vertex_normals)
        # NB - we clip the dot product of the vertex and neighbour normals so as to
        # avoid a div/0 situation as the angle approaches 90 degrees.
        # now clip at 0.5 to avoid being expansive when angles are greater than 60 degrees.
        #alpha = (c_n*n_n).sum(2)/np.maximum((n_n*self.mesh.vertex_normals[:,None,:]).sum(2), 0.5)
        
        n_dot_n = (n_n*self.mesh.vertex_normals[:,None,:]).sum(2)
        alpha = ((c_n*n_n).sum(2))/np.sqrt(2*(np.maximum(n_dot_n, 0) +1))
        
        #print(alpha.shape)
        alpha = (alpha*mask).sum(1)/ms
        
        ### Switch (linearly) between shrinking curvature force and non-shrinking curvature
        # force depending on point influence (how much point-attraction force is acting
        # on the vertex).
        # Low point attraction -> use a shrinking force
        # average/high point attraction -> use a non-shrinking force
        pi = self.mesh.point_influence
        pi = pi #/(pi.sum()/(pi > 0).sum()) # normalise by mean of non-zero entries
        #pi = np.repeat(pi, 3)
        #pi = pi/3.

        #pi = 1

        alpha = alpha*np.minimum(pi**2, 1)
        
        vc = vc + alpha[:,None]*self.mesh.vertex_normals

        vc[ms==0,:] = self.mesh.vertices[ms==0,:]

        return vc


    def _defaults(self, idx=0):
        if idx == 0:
            ## Set defaults for alternative Lfunc formulations where we use 
            ## the centroid of the neighbours, or the modified centroid as a prior
            ## and a weighting matrix (or the identity) as Lfunc/Lhfunc itself
            
            ## If we are using a full Lfunc, we should return 0 (as there is no need for a prior)
            #return 0
            
            ## smoothing moves towards the centroid of the neighbours
            ## this is equivalent to the simple energy discretisation
            #return self._neighbour_centroids().ravel()

            ## smoothing moves towards a point partway between the centroid of the
            ## neighbours, and a point which minimizes the curvature at the neighbours. 
            return self._ncc().ravel()
        
        else:
            # hard coded shrink-wrapping defaults - basically everything propagated in along it's normal
            # this is a purely shinking force.
            
            if self._shrink_def is None:
                #n = self.calculate_normals(self.f).reshape(self.shape)
                n = self.mesh.vertex_normals
                w = self.Ahfunc(np.ones_like(self.points)).reshape(self.shape)
                #print(w.shape)
                w = np.sqrt((w*w).sum(1))

                #print(w.shape, n.shape)

                p = np.maximum(1 - w, 0)[:,None]*n

                self._shrink_def = self.f - 30*p.ravel()

            return self._shrink_def


    def I(self, f):
        # identity operator, used with shrink-wrapping prior
        return f
    
    # def calculate_normals(self, f):
    #     fn = f.reshape(self.shape)
    #     verts = fn[self.faces[self.face_neighbors]]  # (n_vertices, n_neighbors, n_tri_verts, (x,y,z))
    #     v0 = verts[:,:,0,:]
    #     v1 = verts[:,:,1,:]
    #     v2 = verts[:,:,2,:]
    #     t0 = v0-v1
    #     t1 = v2-v1
    #     norms = np.cross(t0,t1,axis=2)

    #     #print(norms, np.any(np.isnan(norms)))

    #     idxs = (self.face_neighbors!=-1)
    #     S = idxs.sum(1)

    #     norms *= idxs[...,None]

    #     #print(norms, np.any(np.isnan(norms)))
    #     # unit_norms = norms/((np.linalg.norm(norms,axis=2)*N[:,None])[...,None])
    #     # return unit_norms.nansum(1).ravel()
    #     norms = norms.sum(1)/S[:,None]
    #     norms /= np.linalg.norm(norms,axis=1)[:,None]
    #     norms[S==0,:] = 0

    #     assert(not np.any(np.isnan(norms)))

    #     return norms.ravel()

    # def Lfuncn(self, f):
    #     """
    #     Minimize difference in normals between a vertex and its neighbors.
    #     """
    #     d = np.zeros_like(f)
    #     norm = self.calculate_normals(f)

    #     for i in range(self.M):
    #         if self.vertex_neighbors[i,0] == -1:
    #             # cheat to find self._vertices['halfedge'] == -1
    #             continue
    #         nn = self.vertex_neighbors[i,:]
    #         S = (nn!=-1).sum()
    #         for n in nn:
    #             if n == -1:
    #                 break
    #             dist = 0
    #             for j in range(self.dims):
    #                 dist += (f[n*self.dims+j] - f[i*self.dims+j])*(f[n*self.dims+j] - f[i*self.dims+j])
    #                 d[i*self.dims+j] += (norm[n*self.dims+j] - norm[i*self.dims+j])
    #             for j in range(self.dims):
    #                 d[i*self.dims+j] /= (S*np.sqrt(dist)+1)
        
    #     assert(not np.any(np.isnan(d)))
        
    #     return d

    # def Lhfuncn(self, f):
    #     # Now we are transposed, so we want to add the neighbors to d in column order
    #     # should be symmetric, unless we change the weighting
    #     d = np.zeros_like(f)
    #     norm = self.calculate_normals(f)

    #     for i in range(self.M):
    #         if self.vertex_neighbors[i,0] == -1:
    #             # cheat to find self._vertices['halfedge'] == -1
    #             continue
    #         nn = self.vertex_neighbors[i,:]
    #         S = (nn!=-1).sum()
    #         for n in nn:
    #             if n == -1:
    #                 break
    #             dist = 0
    #             for j in range(self.dims):
    #                 dist += (f[i*self.dims+j] - f[n*self.dims+j])*(f[i*self.dims+j] - f[n*self.dims+j])
    #                 d[n*self.dims+j] += (norm[i*self.dims+j] - norm[n*self.dims+j])
    #             for j in range(self.dims):
    #                 d[n*self.dims+j] /= (S*np.sqrt(dist)+1)
        
    #     assert(not np.any(np.isnan(d)))
    #     return d

    # # def search(self, data, lams, defaults=None, num_iters=10, weights=1, pos=False, last_step=True):
    # #     self._prev_loopcount = -1
    # #     return TikhonovConjugateGradient.search(self, data, lams, defaults=defaults, 
    # #                                             num_iters=num_iters, weights=weights, 
    # #                                             pos=pos, last_step=last_step)
        
    def start_guess(self, data):
        # since we want to solve ||Af-0|| as part of the
        # equation, we need to pass an array of zeros as
        # data, but guess the verticies for the starting
        # f value
        return self.vertices.copy()

    def _stop_cond(self):
        # Stop if last three test statistcs are within eps of one another
        # (and monotonically decreasing)
        if len(self.tests) < 3:
            return False
        eps = 1e-6
        a, b, c = self.tests[-3:]
        return ((c < b) and (b < a) and (a < eps))
    
    def calc_w(self):
        if self._prev_loopcount < self.loopcount:
            self._prev_loopcount = self.loopcount
            self._shrink_def = None
            return True
        return False


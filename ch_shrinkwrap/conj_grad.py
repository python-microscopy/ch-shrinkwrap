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

class TikhonovConjugateGradient(object):
    """Base N-directional conjugate gradient class for quadratics
    with Tikhonov-regularized terms, implementing a variant of the ICTM algorithm.
    ie. find f such that:
       ||Af-d||^2 + lam0^2||L0(f - fdef0)||^2 + lam1^2||L1(f-fdef1)||| + ...
    is minimised
    
    Note that this is nominally for Gaussian distributed noise, although can be
    adapted by adding a weighting to the misfit term.

    Derived classed should additionally define the following methods:
    AFunc - the forward mapping (computes Af)
    AHFunc - conjugate transpose of forward mapping (computes \bar{A}^T f)
    LFuncs - regularization functions (at least one)
    LHFuncs - conj. transpose of regularization functions

    """
    def __init__(self, *args, **kwargs):
        #allocate some empty lists to track our progress in
        self.tests = []
        self.ress = []
        self.prefs = []

        # List of strings, functions by name
        self.Lfuncs = ["Lfunc"]
        self.Lhfuncs = ["Lhfunc"]

    def start_guess(self, data):
        """starting guess for deconvolution - can be overridden in derived classes
        but the data itself is usually a pretty good guess.
        """
        return data.copy()

    def default_guess(self, default):
        """ guesses for regularization function default solutions, can be overriden
        in derived classes"""
        return default*np.ones(self.f.shape, 'f')

    def searchp(self, args):
        """ convenience function for searching in parallel using processing.Pool.map"""
        self.search(*args)
    
    def search(self, data, lams, defaults=None, num_iters=10, weights=1, pos=False, last_step=True):
        """This is what you actually call to do the deconvolution.
        parameters are:

        Parameters
        ----------
        data : np.array
            the raw data
        lams : list 
            regularisation parameters, list of floats
        defaults : list
            List of float np.arrays indicating default 
            solutions for each regularization parameter
        num_iters : int
            number of iterations
        weights : np.array
            a weighting on the residuals
        pos : bool
            Flag to turn positivity constraints on/off, False default
        """

        if not np.isscalar(weights):
            self.mask = weights > 0
            weights = weights / weights.mean()
        else:
            self.mask = np.isfinite(data.ravel())

        # guess a starting estimate for the object
        # NOTE: start_guess must return a unique object (e.g. a copy() of data)
        self.fs = self.start_guess(data)

        # create a flattened view of our result
        self.f = self.fs.ravel()

        # Assume defaults are 0 for each regularization term if we don't pass any explicitly
        if defaults is None:
            defaults = np.vstack([self.default_guess(0) for x in self.Lfuncs]).T

        #make things 1 dimensional
        data = data.ravel()
        self.res = 0*data

        #number of search directions
        n_smooth = len(self.Lfuncs)
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
        if len(lams) < len(self.Lfuncs):
            print(f"not enough lambdas, defaulting {len(self.Lfuncs)-len(lams)} of them to 0")
            tmp = lams
            lams = [0]*len(self.Lfuncs)
            lams[:len(tmp)] = tmp

        self.loopcount = 0

        while (self.loopcount  < num_iters) and (not self._stop_cond()):
            self.loopcount += 1
            
            # residuals
            self.res[:] = (weights*(data - self.Afunc(self.f)))

            #print ('res:', self.res.reshape(-1, self.dims))

            # search directions
            S[:,0] = self.Ahfunc(self.res)
            for i in range(n_smooth):
                prefs[:,i] = getattr(self, self.Lfuncs[i])(self.f - defaults[:,i]) # residuals
                S[:,i+1] = -1.0*getattr(self, self.Lhfuncs[i])(prefs[:,i])
            
            # check to see if the search directions are orthogonal
            # this can be used as a measure of convergence and a stopping criteria
            test = 1.0
            for pair in search_pairs:
                test -= ((1.0/n_pairs) * abs((S[:,pair[0]]*S[:,pair[1]]).sum()
                        / (np.linalg.norm(S[:,pair[0]])*np.linalg.norm(S[:,pair[1]]))))

            #print & log some statistics
            print(('Test Statistic %f' % (test,)))
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

            #set the current estimate to out new estimate
            self.f[:] = fnew

        return np.real(self.fs)

    def subsearch(self, f0, res, fdefs, Afunc, Lfuncs, lams, S):
        """minimise in subspace - this is the bit which gets called on each iteration
        to work out what the next step is going to be. See Inverse Problems text for details.
        """
        n_search, n_smooth = np.size(S,1), fdefs.shape[1]
        prefs, wpreds = np.zeros(fdefs.shape), np.zeros(n_smooth)
        c0 = (res*res).sum()
        for i in range(n_smooth):
            prefs[:,i] = getattr(self, Lfuncs[i])(f0-fdefs[:,i])
            wpreds[i] = (prefs[i,:]*prefs[i,:]).sum()

        AS = np.zeros((np.size(res), n_search), 'f')
        LS = np.zeros((prefs.shape[0], n_search, n_smooth), 'f')

        for k in range(n_search):
            AS[:,k] = Afunc(S[:,k])[self.mask]
            for i in range(n_smooth):
                LS[:,k,i] = getattr(self, Lfuncs[i])(S[:,k])

        Hc = np.dot(np.transpose(AS), AS)
        Gc = np.dot(np.transpose(AS), res)

        Hw = np.zeros((n_search,n_search,n_smooth))
        Gw = np.zeros((n_search,n_smooth))
        H, G = Hc, Gc
        for i in range(n_smooth):
            ls = LS[:,:,i]
            Hw[:,:,i] = np.dot(np.transpose(ls), ls)
            Gw[:,i] = np.dot(np.transpose(-ls), prefs[:,i])
            l2 = lams[i]*lams[i]
            H += l2*Hw[:,:,i]
            G += l2*Gw[:,i]

        #print(H,G)

        c = np.linalg.solve(H, G)

        #print(c)

        cpred = c0 + np.dot(np.dot(np.transpose(c), Hc), c) - np.dot(np.transpose(c), Gc)
        for i in range(n_smooth):
            wpreds[i] += np.dot(np.dot(np.transpose(c), Hw[:,:,i]), c) - np.dot(np.transpose(c), Gw[:,i])

        fnew = f0 + np.dot(S, c)

        return fnew, cpred, wpreds

    def _stop_cond(self):
        """Optional stopping condition to end deconvolution early."""
        return False

    def Afunc(self, f):
        """ Function that applies A to a vector """
        pass

    def Ahfunc(self, f):
        """ Function that applies conjugate transpose of A to a vector"""
        pass

    def Lfunc(self, f):
        """ Function that applies L (regularization matrix) to a vector"""
        pass

    def Lhfunc(self, f):
        """ Function that applies conjugate transpose of L to a vector"""
        pass

class ShrinkwrapConjGrad(TikhonovConjugateGradient):
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

    def __init__(self, vertices, vertex_neighbors, faces, face_neighbors, points, sigma=None, search_k=200, search_rad=100):
        TikhonovConjugateGradient.__init__(self)
        self.Lfuncs, self.Lhfuncs = ["Lfunc3"], ["Lhfunc3"]
        self.vertices, self.vertex_neighbors, self.sigma = vertices, vertex_neighbors, sigma
        self.faces, self.face_neighbors = faces, face_neighbors
        self.points = points
        self.search_k = search_k
        self.search_rad = search_rad
        self._prev_loopcount = -1

    def search(self, data, lams, defaults=None, num_iters=10, weights=1, pos=False, last_step=True):
        """Custom search to add weighting to res
        """

        self._prev_loopcount = -1

        if not np.isscalar(weights):
            self.mask = weights > 0
            weights = weights / weights.mean()
        else:
            self.mask = np.isfinite(data.ravel())

        # guess a starting estimate for the object
        # NOTE: start_guess must return a unique object (e.g. a copy() of data)
        self.fs = self.start_guess(data)

        # create a flattened view of our result
        self.f = self.fs.ravel()

        # Assume defaults are 0 for each regularization term if we don't pass any explicitly
        if defaults is None:
            defaults = np.vstack([self.default_guess(0) for x in self.Lfuncs]).T

        #make things 1 dimensional
        data = data.ravel()
        self.res = 0*data

        #number of search directions
        n_smooth = len(self.Lfuncs)
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
        if len(lams) < len(self.Lfuncs):
            print(f"not enough lambdas, defaulting {len(self.Lfuncs)-len(lams)} of them to 0")
            tmp = lams
            lams = [0]*len(self.Lfuncs)
            lams[:len(tmp)] = tmp

        self.loopcount = 0

        while (self.loopcount  < num_iters) and (not self._stop_cond()):
            self.loopcount += 1
            
            # residuals
            self.res[:] = (weights*(data - self.Afunc(self.f)))

            # weight the residuals based on distance
            # /8 = 2 * 2^2 for weighting within 2*sigma of the point
            # w = np.exp(-(self.d.ravel()**2)*((weights/2)**2)) + 1/(self.d.ravel()**2+1)
            #w = 0.5-np.arctan(self.d.ravel()**2-2.0/weights**2)/np.pi
            w = 1.0/(self.d.ravel()*weights/2.0+1)
            # w = np.exp(-(self.d.ravel()*weights/2)**2)
            # w = 1.0/np.log((2*self.d.ravel()/weights)**2+np.exp(1))
            # print("WEIGHTING")
            # print(w)
            self.res *= w

            #print ('res:', self.res.reshape(-1, self.dims))

            # search directions
            S[:,0] = self.Ahfunc(self.res)
            for i in range(n_smooth):
                prefs[:,i] = getattr(self, self.Lfuncs[i])(self.f - defaults[:,i]) # residuals
                S[:,i+1] = -1.0*getattr(self, self.Lhfuncs[i])(prefs[:,i])
            
            # check to see if the search directions are orthogonal
            # this can be used as a measure of convergence and a stopping criteria
            test = 1.0
            for pair in search_pairs:
                test -= ((1.0/n_pairs) * abs((S[:,pair[0]]*S[:,pair[1]]).sum()
                        / (np.linalg.norm(S[:,pair[0]])*np.linalg.norm(S[:,pair[1]]))))

            #print & log some statistics
            print(('Test Statistic %f' % (test,)))
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

            #set the current estimate to out new estimate
            self.f[:] = fnew

        return np.real(self.fs)
        
    def _compute_weight_matrix(self, f, w=0.95, shield_sigma=20):
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

        # Create a list of face centroids for search
        face_centers = fv[self._faces].mean(1)

        # Construct a kdtree over the face centers
        tree = scipy.spatial.cKDTree(face_centers)

        # Get k closet face centroids for each point
        dmean, _faces = tree.query(self.points, k=1)
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

        w = 1.0/np.maximum(d, 1e-6)

        w = w/w.sum(1)[:,None]

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
            
            for i in range(3):
                d[v_idx[:,i], :] += (w[:,i][:,None])*fv 
                # d[v_idx2[:,i], :] += (w2[:,i][:,None])*fv 

        
        assert(not np.any(np.isnan(d)))
        #print('ah:',d)
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
        d = np.zeros_like(f)
        conj_grad_utils.c_shrinkwrap_lw_func(np.ascontiguousarray(f), self.vertex_neighbors, self.f, d, self.dims, self.points.shape[0], self.M, self.N)
        
        assert(not np.any(np.isnan(d)))
        return d

    def Lhfunc3(self, f):
        d = np.zeros_like(f)
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

    def calculate_normals(self, f):
        fn = f.reshape(self.shape)
        verts = fn[self.faces[self.face_neighbors]]  # (n_vertices, n_neighbors, n_tri_verts, (x,y,z))
        v0 = verts[:,:,0,:]
        v1 = verts[:,:,1,:]
        v2 = verts[:,:,2,:]
        t0 = v0-v1
        t1 = v2-v1
        norms = np.cross(t0,t1,axis=2)

        #print(norms, np.any(np.isnan(norms)))

        idxs = (self.face_neighbors!=-1)
        S = idxs.sum(1)

        norms *= idxs[...,None]

        print(norms, np.any(np.isnan(norms)))
        # unit_norms = norms/((np.linalg.norm(norms,axis=2)*N[:,None])[...,None])
        # return unit_norms.nansum(1).ravel()
        norms = norms.sum(1)/S[:,None]
        norms /= np.linalg.norm(norms,axis=1)[:,None]
        norms[S==0,:] = 0

        assert(not np.any(np.isnan(norms)))

        return norms.ravel()

    def Lfuncn(self, f):
        """
        Minimize difference in normals between a vertex and its neighbors.
        """
        d = np.zeros_like(f)
        norm = self.calculate_normals(f)

        for i in range(self.M):
            if self.vertex_neighbors[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            nn = self.vertex_neighbors[i,:]
            S = (nn!=-1).sum()
            for n in nn:
                if n == -1:
                    break
                dist = 0
                for j in range(self.dims):
                    dist += (f[n*self.dims+j] - f[i*self.dims+j])*(f[n*self.dims+j] - f[i*self.dims+j])
                    d[i*self.dims+j] += (norm[n*self.dims+j] - norm[i*self.dims+j])
                for j in range(self.dims):
                    d[i*self.dims+j] /= (S*np.sqrt(dist)+1)
        
        assert(not np.any(np.isnan(d)))
        
        return d

    def Lhfuncn(self, f):
        # Now we are transposed, so we want to add the neighbors to d in column order
        # should be symmetric, unless we change the weighting
        d = np.zeros_like(f)
        norm = self.calculate_normals(f)

        for i in range(self.M):
            if self.vertex_neighbors[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            nn = self.vertex_neighbors[i,:]
            S = (nn!=-1).sum()
            for n in nn:
                if n == -1:
                    break
                dist = 0
                for j in range(self.dims):
                    dist += (f[i*self.dims+j] - f[n*self.dims+j])*(f[i*self.dims+j] - f[n*self.dims+j])
                    d[n*self.dims+j] += (norm[i*self.dims+j] - norm[n*self.dims+j])
                for j in range(self.dims):
                    d[n*self.dims+j] /= (S*np.sqrt(dist)+1)
        
        assert(not np.any(np.isnan(d)))
        return d

    # def search(self, data, lams, defaults=None, num_iters=10, weights=1, pos=False, last_step=True):
    #     self._prev_loopcount = -1
    #     return TikhonovConjugateGradient.search(self, data, lams, defaults=defaults, 
    #                                             num_iters=num_iters, weights=weights, 
    #                                             pos=pos, last_step=last_step)
        
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
            return True
        return False

class SkeletonConjGrad(TikhonovConjugateGradient):
    """
    Collapse surface to a skeleton. Note this requries last_step=False when
    calling search().

    Tagliasacchi, Andrea, Ibraheem Alhashim, Matt Olson, and Hao Zhang. 
    "Mean Curvature Skeletons." Computer Graphics Forum 31, no. 5 
    (August 2012): 1735–44. https://doi.org/10.1111/j.1467-8659.2012.03178.x.
    """
    _vertices = None
    _vertex_neighbors = None
    _prev_vertices = None
        
    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices):
        self._vertices = vertices
        self.M = vertices.shape[0]
        self.dims = vertices.shape[1]
        self.shape = vertices.shape  # hack
        self._on_deck_vertices = vertices.copy().ravel()
        self._prev_vertices = vertices.copy().ravel()+0.01*self._vertex_normals.ravel()
        
    @property
    def vertex_normals(self):
        return self._vertex_normals
    
    @vertex_normals.setter
    def vertex_normals(self, normals):
        self._vertex_normals = normals
        
    @property
    def vertex_neighbors(self):
        return self._vertex_neighbors
    
    @vertex_neighbors.setter
    def vertex_neighbors(self, neighbors):
        self._vertex_neighbors = neighbors
        self.N = self._vertex_neighbors.shape[1]
        
    def Afunc(self, f):
        """
        Minimize distance between a vertex and the centroid of its neighbors.
        """
        # note that f is raveled, by default in C order so 
        # f = [v0x, v0y, v0z, v1x, v1y, v1z, ...] where ij is vertex i, dimension j
        d = np.zeros_like(f)
        for i in range(self.M):
            if self.vertex_neighbors[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            for j in range(self.dims):
                nn = self.vertex_neighbors[i,:]
                N = (nn!=-1).sum()
                for n in nn:
                    if n == -1:
                        break
                    d[i*self.dims+j] += (f[n*self.dims+j] - f[i*self.dims+j])/N
        return d
    
    def Ahfunc(self, f):
        # Now we are transposed, so we want to add the neighbors to d in column order
        # should be symmetric, unless we change the weighting
        d = np.zeros_like(f)
        for i in range(self.M):
            if self.vertex_neighbors[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            for j in range(self.dims):
                nn = self.vertex_neighbors[i,:]
                N = (nn!=-1).sum()
                for n in nn:
                    if n == -1:
                        break
                    d[n*self.dims+j] += (f[i*self.dims+j] - f[n*self.dims+j])/N
        return d
    
    def Lfunc(self, f):
        """
        Velocity term.
        """
        if self._updated_loopcount():
            self._prev_vertices = self._on_deck_vertices
            self._on_deck_vertices = self.f.copy()
        idxs = np.repeat(self.vertex_neighbors[:,0]==-1,3)
        val = (f - self._prev_vertices)
        val[idxs] = 0
        return val
    
    def Lhfunc(self, f):
        idxs = np.repeat(self.vertex_neighbors[:,0]==-1,3)
        val = f
        val[idxs] = 0
        return f
    
    def Mfunc(self, f):
        """
        Distance to medial axis.
        """
        fr = f.reshape(self.shape)
        #print(fr)
        _, nearest_pole = self._neg_vor_poles_tree.query(fr,1)

        # none of the missing voronoi poles will matter to the final result,
        # as they are from -1 halfedge vertices, but they will throw an error
        # as such, replace them with something "valid"
        idxs = (self.vertex_neighbors[:,0]==-1) | (nearest_pole == self._neg_vor_poles.shape[0])
        nearest_pole[idxs] = 0

        #print(nearest_pole)
        #print(self._neg_vor_poles[nearest_pole,:])
        val = (self._neg_vor_poles[nearest_pole,:]-fr)
        val[idxs,:] = 0
        return val.ravel()
    
    def Mhfunc(self, f):
        # fr = f.reshape(self.shape)
        # _, nearest_pole = self._neg_vor_poles_tree.query(fr,1)
        idxs = np.repeat(self.vertex_neighbors[:,0]==-1,3)
        val = f
        val[idxs] = 0
        return f
    
    def __init__(self, vertices, vertex_normals, neighbors, *args, **kwargs):
        TikhonovConjugateGradient.__init__(self, *args, **kwargs)
        self.Lfuncs = ["Lfunc", "Mfunc"]
        self.Lhfuncs = ["Lhfunc", "Mhfunc"]
        self.vertex_neighbors, self.vertex_normals, self.vertices = neighbors, vertex_normals, vertices
        self._prev_loopcount = 1
        self._vor = scipy.spatial.Voronoi(self._vertices)
        _, pn = voronoi_poles(self._vor, self.vertex_normals)
        self._neg_vor_poles = self._vor.vertices[pn[pn!=-1]]
        if kwargs.get('mesh'):
            from .delaunay_utils import clean_neg_voronoi_poles
            self._neg_vor_poles = clean_neg_voronoi_poles(kwargs.get('mesh'),self._neg_vor_poles)
        self._neg_vor_poles_tree = scipy.spatial.cKDTree(self._neg_vor_poles)

    def search(self, data, lams, defaults=None, num_iters=10, weights=1, pos=False, last_step=False):
        self._prev_loopcount = 1
        return TikhonovConjugateGradient.search(self, data, lams, defaults=defaults, 
                                                num_iters=num_iters, weights=weights, 
                                                pos=pos, last_step=last_step)
        
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
    
    def _updated_loopcount(self):
        if self._prev_loopcount < self.loopcount:
            print("calculate")
            self._prev_loopcount = self.loopcount
            return True
        return False
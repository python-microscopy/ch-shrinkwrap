#!/usr/bin/python

###################################
# Based on PYME.Deconv.dec dec.py
###################################

from .delaunay_utils import voronoi_poles

import numpy as np
import scipy.spatial

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

            # search directions
            S[:,0] = self.Ahfunc(self.res)
            for i in np.arange(n_smooth):
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
    _neighbors = None
    _sigma = None
    _search_rad = 100
    
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
    def neighbors(self):
        return self._neighbors
    
    @neighbors.setter
    def neighbors(self, neighbors):
        self.n = neighbors
        self.N = self.n.shape[1]
    
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

    def __init__(self, vertices, neighbors, points, sigma=None, search_k=200, search_rad=100):
        TikhonovConjugateGradient.__init__(self)
        self.vertices, self.neighbors, self.sigma = vertices, neighbors, sigma
        self.points = points
        self.search_k = search_k
        self.search_rad = search_rad
        self._prev_loopcount = -1
        
    def _compute_weight_matrix(self, f, w=0.95, shield_sigma=20):
        """
        Construct an n_vertices x n_points matrix.
        """
        dd = np.zeros((self.M,self.points.shape[0]))
        fv = f.reshape(-1,self.dims)
        
        for i in np.arange(self.M):
            if self.n[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            # Grab all neighbors within search_rad or the nearest search_k neighbors
            # if there are no neighbors within search_rad
            neighbors = self._tree.query_ball_point(fv[i,:], self.search_rad)
            if len(neighbors) == 0:
                _, neighbors = self._tree.query(fv[i,:], self.search_k)
            for k in neighbors:  # np.arange(self.points.shape[0]):
                for j in np.arange(self.dims):
                    dik = (f[i*self.dims+j] - self.points[k,j])
                    dd[i,k] += dik*dik
                    
                if dd[i,k] > 0:
                    #dd[i,k] = self.sigma[k]*self.sigma[k]/dd[i,k]
                    dd[i,k] = (1.0/dd[i,k])*np.exp(-dd[i,k]/(2*shield_sigma*shield_sigma))

        # normalize s.t. the sum of the distances from each point to every other vertex = 1
#         dd2 = np.copy(dd)
        ds = dd.sum(0)                 
        dd[:,ds>0] /= ds[None,ds>0]
        
        # normalize s.t. the sum of the distances from each vertex to every other point = 1
#         ds = dd2.sum(1)
#         dd2[ds>0,:] /= ds[ds>0,None]
                        
        return dd  # , dd2
    
    
    def Afunc(self, f):
        """
        Create a map of which (weighted average) vertex each point should drive toward.
        
        f is a set of vertices
        d is the weighted sum of all of the vertices indicating the closest vertex to each point
        """
        if self.calc_w():
            self.w = self._compute_weight_matrix(self.f)
        
        # Compute the distance between the vertex and all the self.pts, weighted
        # by distance to the vertex, sigma, etc.
        d = np.zeros_like(self.points.ravel())
        for i in np.arange(self.M):
            if self.n[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            iv = np.array([self.f[i*self.dims+j] for j in range(self.dims)])
            neighbors = self._tree.query_ball_point(iv, self.search_rad)
            if len(neighbors) == 0:
                _, neighbors = self._tree.query(iv, self.search_k)
            #print(f"# neighbors: {len(neighbors)}")
            for k in neighbors:  # np.arange(self.points.shape[0]):
                for j in np.arange(self.dims):
                    d[k*self.dims+j] += f[i*self.dims+j]*self.w[i,k]

        return d
    
    def Ahfunc(self, f):
        """
        Map each distance between a point and its closest (weighted average) vertex to
        a new vertex position.
        
        f is a set of points
        d is the weighted sum of all of the points indicating the closest point to each vertex
        """
        
        d = np.zeros(self.M*self.dims)
        
        for i in np.arange(self.M):
            if self.n[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            iv = np.array([self.f[i*self.dims+j] for j in range(self.dims)])
            neighbors = self._tree.query_ball_point(iv, self.search_rad)
            if len(neighbors) == 0:
                _, neighbors = self._tree.query(iv, self.search_k)
            #print(f"# neighbors: {len(neighbors)}")
            for k in neighbors:  # np.arange(self.points.shape[0]):
                for j in np.arange(self.dims):
                    d[i*self.dims+j] += f[k*self.dims+j]*self.w[i,k]

        return d

    def Lfunc(self, f):
        """
        Minimize distance between a vertex and the centroid of its neighbors.
        """
        # note that f is raveled, by default in C order so 
        # f = [v0x, v0y, v0z, v1x, v1y, v1z, ...] where ij is vertex i, dimension j
        d = np.zeros_like(f)
        for i in np.arange(self.M):
            if self.n[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            for j in np.arange(self.dims):
                nn = self.n[i,:]
                N = len(nn)
                for n in nn:
                    if n == -1:
                        break
                    d[i*self.dims+j] += (f[n*self.dims+j] - f[i*self.dims+j])/N
        return d
    
    def Lhfunc(self, f):
        # Now we are transposed, so we want to add the neighbors to d in column order
        # should be symmetric, unless we change the weighting
        d = np.zeros_like(f)
        for i in np.arange(self.M):
            if self.n[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            for j in np.arange(self.dims):
                nn = self.n[i,:]
                N = len(nn)
                for n in nn:
                    if n == -1:
                        break
                    d[n*self.dims+j] += (f[i*self.dims+j] - f[n*self.dims+j])/N
        return d

    def search(self, data, lams, defaults=None, num_iters=10, weights=1, pos=False, last_step=True):
        self._prev_loopcount = -1
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
    _neighbors = None
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
        self._prev_vertices = vertices.copy().ravel()+0.01*np.random.randn(*vertices.shape).ravel()
        
    @property
    def vertex_normals(self):
        return self._vertex_normals
    
    @vertex_normals.setter
    def vertex_normals(self, normals):
        self._vertex_normals = normals
        
    @property
    def neighbors(self):
        return self._neighbors
    
    @neighbors.setter
    def neighbors(self, neighbors):
        self.n = neighbors
        self.N = self.n.shape[1]
        
    def Afunc(self, f):
        """
        Minimize distance between a vertex and the centroid of its neighbors.
        """
        # note that f is raveled, by default in C order so 
        # f = [v0x, v0y, v0z, v1x, v1y, v1z, ...] where ij is vertex i, dimension j
        d = np.zeros_like(f)
        for i in np.arange(self.M):
            if self.n[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            for j in np.arange(self.dims):
                nn = self.n[i,:]
                N = len(nn)
                for n in nn:
                    if n == -1:
                        break
                    d[i*self.dims+j] += (f[n*self.dims+j] - f[i*self.dims+j])/N
        return d
    
    def Ahfunc(self, f):
        # Now we are transposed, so we want to add the neighbors to d in column order
        # should be symmetric, unless we change the weighting
        d = np.zeros_like(f)
        for i in np.arange(self.M):
            if self.n[i,0] == -1:
                # cheat to find self._vertices['halfedge'] == -1
                continue
            for j in np.arange(self.dims):
                nn = self.n[i,:]
                N = len(nn)
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
        idxs = np.repeat(self.n[:,0]==-1,3)
        val = (f - self._prev_vertices)
        val[idxs] = 0
        return val
    
    def Lhfunc(self, f):
        idxs = np.repeat(self.n[:,0]==-1,3)
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
        idxs = (self.n[:,0]==-1) | (nearest_pole == self._neg_vor_poles.shape[0])
        nearest_pole[idxs] = 0

        #print(nearest_pole)
        #print(self._neg_vor_poles[nearest_pole,:])
        val = (self._neg_vor_poles[nearest_pole,:]-fr)
        val[idxs,:] = 0
        return val.ravel()
    
    def Mhfunc(self, f):
        # fr = f.reshape(self.shape)
        # _, nearest_pole = self._neg_vor_poles_tree.query(fr,1)
        idxs = np.repeat(self.n[:,0]==-1,3)
        val = f
        val[idxs] = 0
        return f
    
    def __init__(self, vertices, vertex_normals, neighbors, *args, **kwargs):
        TikhonovConjugateGradient.__init__(self, *args, **kwargs)
        self.Lfuncs = ["Lfunc", "Mfunc"]
        self.Lhfuncs = ["Lhfunc", "Mhfunc"]
        self.vertices, self.neighbors, self.vertex_normals = vertices, neighbors, vertex_normals
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
#!/usr/bin/python

###################################
# Based on PYME.Deconv.dec dec.py
###################################

from .delaunay_utils import voronoi_poles

cimport numpy as np
import numpy as np
import scipy.spatial
import cython

cdef class TikhonovConjugateGradient(object):
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
    cdef public list tests
    cdef public list ress
    cdef public list prefs
    cdef public list Lfuncs
    cdef public list Lhfuncs
    cdef public np.ndarray f
    cdef np.ndarray fs
    cdef public np.ndarray res
    cdef int loopcount
    cdef np.ndarray mask
    cdef public object cpred
    cdef public object wpreds
    cdef float * _cf
    cdef float * _cres
    cdef float * _cdefaults
    cdef float * _cprefs
    cdef float * _cS

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

    cdef _set_cf(self, np.ndarray f):
        self._cf = <float *> np.PyArray_DATA(f)

    cdef _set_cS(self, np.ndarray s):
        self._cS = <float *> np.PyArray_DATA(s)

    cdef _set_cf_minus_defaults(self, np.ndarray f):
        self._cf_minus_defaults = <float *> np.PyArray_DATA(f)

    cdef _set_cres(self, np.ndarray res):
        self._cres = <float *> np.PyArray_DATA(res)

    cdef _set_cdefaults(self, np.ndarray defaults):
        self._cdefaults = <float *> np.PyArray_DATA(_cdefaults)
    
    cdef _set_cprefs(self, np.ndarray prefs):
        self._cprefs = <float *> np.PyArray_DATA(prefs)
    
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
        self._set_cf(self.f)

        # Assume defaults are 0 for each regularization term if we don't pass any explicitly
        if defaults is None:
            defaults = np.vstack([self.default_guess(0) for x in self.Lfuncs]).T

        _flat_defaults = defaults.ravel()
        self._cdefaults(_flat_defaults)

        _cf_minus_defaults = np.zeros(defaults.shape[0], dtype=float)
        self._set_cf_minus_defaults(_cf_minus_defaults)

        #make things 1 dimensional
        data = data.ravel()
        self.res = 0*data
        self._set_cres(self.res)

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
        _flat_prefs = prefs.ravel()
        self._set_cprefs(_flat_prefs)

        #initial search directions
        S = np.zeros((np.size(self.f), s_size), 'f')
        _flat_S = S.ravel()
        self._set_cS(_flat_S)

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
            self.res[:] = (weights*(data - self.Afunc(&(self._cf[0]))))

            # search directions
            S[:,0] = self.Ahfunc(&(self._cres[0]))
            for i in np.arange(n_smooth):
                for j in np.arange(defaults.shape[0]):
                    self._cf_minus_defaults = self._cf[j] - self._cdefaults[i*defaults.shape[0]+j]
                prefs[:,i] = getattr(self, self.Lfuncs[i])(&(self._cf_minus_defaults[0]))) # residuals
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

cdef class SkeletonConjGrad(TikhonovConjugateGradient):
    """
    Collapse surface to a skeleton. Note this requries last_step=False when
    calling search().

    Tagliasacchi, Andrea, Ibraheem Alhashim, Matt Olson, and Hao Zhang. 
    "Mean Curvature Skeletons." Computer Graphics Forum 31, no. 5 
    (August 2012): 1735–44. https://doi.org/10.1111/j.1467-8659.2012.03178.x.
    """
    cdef np.ndarray _vertices
    cdef np.ndarray _neighbors
    cdef np.ndarray _prev_vertices
    cdef int M
    cdef int dims
    cdef tuple shape
    cdef np.ndarray _on_deck_vertices
    cdef object _vor
    cdef np.ndarray _vertex_normals
    cdef np.ndarray n
    cdef int N
    cdef np.ndarray _neg_vor_poles
    cdef object _neg_vor_poles_tree
    cdef int _prev_loopcount
        
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
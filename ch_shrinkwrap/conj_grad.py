#!/usr/bin/python

###################################
# Based on PYME.Deconv.dec dec.py
###################################

import numpy as np

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
    def __init__(self):
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
        return np.zeros(self.f.shape, 'f')

    def searchp(self, args):
        """ convenience function for searching in parallel using processing.Pool.map"""
        self.search(*args)
    
    def search(self, data, lams, defaults=None, num_iters=10, weights=1, pos=False):
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

        #guess a starting estimate for the object
        self.fs = self.start_guess(data)

        # create a view of the correct shape for our result 
        self.f = self.fs.ravel()

        #make things 1 dimensional
        data = data.ravel()
        self.res = 0*data

        #number of search directions
        n_smooth = len(self.Lfuncs)
        n_search = n_smooth+1
        s_size = n_search+1

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
        if type(lams) is float:
            lams = [lams]

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
                S[:,i+1] = -getattr(self, self.Lhfuncs[i])(prefs[:,i])
            
            # check to see if the search directions are orthogonal
            # this can be used as a measure of convergence and a stopping criteria
            test = 1.0
            for pair in search_pairs:
                test -= (1.0/n_pairs)*abs((S[:,pair[0]]*S[:,pair[1]]).sum()/(np.linalg.norm(S[:,pair[0]])*np.linalg.norm(S[:,pair[1]])))

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

        c = np.linalg.solve(H, G)

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
        
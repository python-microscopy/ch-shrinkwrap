from PYME.Deconv.dec import DeconvMappingBase, ICTMDeconvolution

import numpy as np
import scipy.spatial

class MappingCurvature(DeconvMappingBase):
    _search_k = 200
    _points = None
    _vertices = None
    _neighbors = None
    _sigma = None
    
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
            _, neighbors = self._tree.query(iv, self.search_k)
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
            _, neighbors = self._tree.query(iv, self.search_k)
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
            for j in np.arange(self.dims):
                nn = self.n[i,:]
                N = len(nn)
                for n in nn:
                    if n == -1:
                        break
                    d[n*self.dims+j] += (f[i*self.dims+j] - f[n*self.dims+j])/N
        return d
    
    def calc_w(self):
        return True
    
class dec_curv(ICTMDeconvolution, MappingCurvature):
    """
    Apply ICTM deconvolution to wrap self.vertices to self.points, subject to a curvature
    constraint.
    
    Parameters
    ----------
    vertices : np.array
        N x dimension set of vertices
    neighbors : np.array
        N x # neighbors set of vertex neighbors, assumed to be connected by an edge
    sigma : np.array
        Size N array of uncertainties in vertex position, optional
    points: np.array
    """
    def __init__(self, vertices, neighbors, points, sigma=None, search_k=200, *args, **kwargs):
        ICTMDeconvolution.__init__(self, *args, **kwargs)
        self.vertices, self.neighbors, self.sigma = vertices, neighbors, sigma
        self.points = points
        self.search_k = search_k
        self._prev_loopcount = -1
        
    def startGuess(self, data):
        # since we want to solve ||Af-0|| as part of the
        # equation, we need to pass an array of zeros as
        # data, but guess the verticies for the starting
        # f value
        return self.vertices

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

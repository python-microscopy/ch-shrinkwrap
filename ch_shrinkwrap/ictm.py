from PYME.Deconv.dec import DeconvMappingBase, ICTMDeconvolution

import numpy as np
import scipy.spatial

class MappingCurvature(DeconvMappingBase):
    def prep(self):
        pass
    
    def set_vertices_neighbors(self, vertices, neighbors, sigma=None):
        self.vertices = vertices
        self.M = vertices.shape[0]
        self.dims = vertices.shape[1]
        self.n = neighbors
        self.N = self.n.shape[1]
        self.shape = vertices.shape  # hack
        self.sigma = sigma
        
        assert(self.n.shape[0] == self.shape[0])
        
    def set_points(self, pts):
        self.points = pts
        # Compute a KDTree on points
        self._tree = scipy.spatial.cKDTree(pts)
        
    def _compute_weight_matrix(self, f, w=0.95, shield_sigma=20, search_k=200):
        """
        Construct an n_vertices x n_points matrix.
        """
        dd = np.zeros((self.M,self.points.shape[0]))
        
        for i in np.arange(self.M):
            _, neighbors = self._tree.query(self.vertices[i,:], min(search_k, self.points.shape[0]))
            for k in neighbors: # np.arange(self.points.shape[0]):
                for j in np.arange(self.dims):
                    dik = (f[i*self.dims+j] - self.points[k,j])
                    dd[i,k] += dik*dik
                    
                if dd[i,k] > 0:
                    #dd[i,k] = self.sigma[k]*self.sigma[k]/dd[i,k]
                    dd[i,k] = (1.0/dd[i,k])*np.exp(-dd[i,k]/(2*shield_sigma*shield_sigma))

#         shielded_by, shielded = np.where(dd < shield_sigma)
        
#         for i, k in zip(shielded_by, shielded):
#             dd[:,k] = 0
#             dd[i,k] = 1
        
        ds = dd.sum(0)                 
        dd[:,ds>0] /= ds[None,ds>0]
        
#         ds = dd.sum(1)
#         dd[ds >0, :] /= ds[ds>0, None]
                        
        return dd
    
    
    def Afunc(self, f, search_k=200):
        """
        Map each vertex to a weighted distance between it and all of the points.
        
        Here f represents the target vertices,
        d is the mismatch between a point and weighted sum of all of the vertices
        """
        self.w = self._compute_weight_matrix(self.f)
        
        # Compute the distance between the vertex and all the self.pts, weighted
        # by distance to the vertex, sigma, etc.
        d = np.zeros_like(self.points.ravel())
        for i in np.arange(self.M):
            _, neighbors = self._tree.query(self.vertices[i,:], min(search_k, self.points.shape[0]))
            for k in neighbors: # np.arange(self.points.shape[0]):
                for j in np.arange(self.dims):
                    d[k*self.dims+j] += f[i*self.dims+j]*self.w[i,k]

        return d
    
    def Ahfunc(self, f, search_k=200):
        """
        Map each point to a weighted distance between it and all of the vertices
        
        f is the mismatch between a point and weighted sum of all of the vertices
        d is the mismatch between a vertex and the weighted sum of points
        """
        
        d = np.zeros(self.M*self.dims)
        
        for i in np.arange(self.M):
            _, neighbors = self._tree.query(self.vertices[i,:], min(search_k, self.points.shape[0]))
            for k in neighbors: # np.arange(self.points.shape[0]):
                for j in np.arange(self.dims):
                    d[i*self.dims+j] += - f[k*self.dims+j]*self.w[i,k]

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
                for n in self.n[i,:]:
                    if n == -1:
                        break
                    d[i*self.dims+j] += (f[n*self.dims+j] - f[i*self.dims+j])/self.N
        return d
    
    def Lhfunc(self, f):
        # Now we are transposed, so we want to add the neighbors to d in column order
        # should be symmetric, unless we change the weighting
        d = np.zeros_like(f)
        for i in np.arange(self.M):
            for j in np.arange(self.dims):
                for n in self.n[i,:]:
                    if n == -1:
                        break
                    d[n*self.dims+j] += (f[i*self.dims+j] - f[n*self.dims+j])/self.N
        return d
    
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
        Size N a rray of uncertainties in vertex position, optional
    gamma : float
        Target mean edge length (distance between neighboring vertices). If None,
        defaults to average of current edge lengths, optional
    points: np.array
    """
    def __init__(self, vertices, neighbors, sigma=None, gamma=None, points=None, *args, **kwargs):
        ICTMDeconvolution.__init__(self, *args, **kwargs)
        self.set_vertices_neighbors(vertices,neighbors,sigma)
        self.set_points(points)
        self.gamma = gamma
        
    def startGuess(self, data):
        # since we want to solve ||Af-0|| as part of the
        # equation, we need to pass an array of zeros as
        # data, but guess the verticies for the starting
        # f value
        return self.vertices

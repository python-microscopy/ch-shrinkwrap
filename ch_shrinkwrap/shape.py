import numpy as np
from ch_shrinkwrap import sdf_octree

class Shape:
    def __init__(self, **kwargs):
        """
        A general class for generating geometric shapes, and tools to compare 
        these shapes with externally-generated approximations. All Shapes are
        based in constructive solid geometry.

        SDFs from http://iquilezles.org/www/articles/distfunctions/distfunctions.htm.
        """
        self._density = None  # Shape density
        self._points = None  # Point queue representing Shape
        self._radius = None  # maximum diameter of the shape/2
        
        for k, v in kwargs.items():
            self.__setattr__(k, v)
            
    @property
    def surface_area(self):
        raise NotImplementedError('Implemented in a derived class.')
        
    @property
    def volume(self):
        raise NotImplementedError('Implemented in a derived class.')
        
    def sdf(self, points):
        """ Signed distance function """
        raise NotImplementedError('Implemented in a derived class')
        
    def _noise_model(self, shape, model='poisson'):
        """
        Noise model for point jitter. Currently distributed as N(0,1).
        
        Parameters
        ----------
            shape : tupule
                Shape of additional noise to return (presumaly N_points x 3).
        """
        models = ['gaussian', 'poisson']
        if model not in models:
            print('Unknown noise model {}, defaulting to poisson.'.format(model))
            model = 'poisson'
        
        if model is 'gaussian':
            return np.random.randn(*shape)
        
        if model is 'poisson':
            return np.random.poisson(1,shape)
        
    def _noise_model(self, shape):
        """
        Noise model for point jitter. Currently distributed as N(0,1).
        
        Parameters
        ----------
            shape : tupule
                Shape of additional noise to return (presumaly N_points x 3).
        """
        return np.random.randn(*shape)
    
    def points(self, density=1, p=0.1, noise=0, resample=False, eps=0.001):
        """
        Monte-Carlo sampling of uniform points on the Shape surface.
        
        Parameters
        ----------
            density : float
                Fluorophores per nm.
            p : float
                Likelihood that a fluorophore is detected.
            noise : float
                Noise distributed as defined by self._noise_model.
            resample : bool
                Redo point sampling at each function call.
        """
        if resample or (self._points is None) or (self._density != density):
            self._density = density
            # rp = 2*self._radius*(np.random.rand(int(np.round((self._radius*density)**3)), 3) - 0.5)
            r = 2.0*self._radius
            ot = sdf_octree.SDFOctree([-1.0*r, r, -1.0*r, r, -1.0*r, r], self.sdf, self._density, eps)
            points_raw = ot.points()
            # points_raw = rp[np.abs(self.sdf(rp)) < eps]
            if (noise > 0):
                points_raw += noise*self._noise_model((points_raw.shape[0], 3))
            pr = np.random.rand(points_raw.shape[0])
            self._points = points_raw[pr < p]  # Make a Monte-Carlo decision
        return self._points
    
    def surface_res(self, points):
        return np.sum(self.sdf(points)**2)
        
class Sphere(Shape):
    def __init__(self, radius=1, **kwargs):
        """
        Parameters
        ----------
            radius : float
                Radius of the sphere
        """
        super(Sphere, self).__init__(**kwargs)
        self._radius = radius
        
    @property
    def surface_area(self):
        return 4*np.pi*self._radius**2
    
    @property
    def volume(self):
        return (4.0/3.0)*np.pi*self._radius**3
    
    def sdf(self, p):
        if len(p.shape) > 1:
            return np.sqrt(np.sum(p*p, axis=1)) - self._radius
        return np.sqrt(np.sum(p*p)) - self._radius

class Ellipsoid(Shape):
    def __init__(self, a=1, b=1, c=2, **kwargs):
        super(Ellipsoid, self).__init__(**kwargs)
        self._a = a
        self._b = b
        self._c = c
        self._radius = np.max([a,b,c])
        
    @property
    def surface_area(self):
        p = 1.6075
        return 4*np.pi*((self._a**p*self._b**p+
                         self._a**p*self._c**p+
                         self._b**p*self._c**p)/3)**(1/p)
    
    @property
    def volume(self):
        return (4.0/3.0)*np.pi*self._radius**3*self._a*self._b*self._c
        
    def sdf(self, p):
        r = np.array([self._a,self._b,self._c])
        pr = p/r
        prr = pr/r
        if len(p.shape) > 1:
            k0 = np.sqrt(np.sum(pr*pr,axis=1))
            k1 = np.sqrt(np.sum(prr*prr,axis=1))
            return k0*(k0-1.0)/k1
        k0 = np.sqrt(np.sum(pr*pr))
        k1 = np.sqrt(np.sum(prr*prr))
        return k0*(k0-1.0)/k1

class Torus(Shape):
    def __init__(self, radius=2, r=0.05, **kwargs):
        super(Torus, self).__init__(**kwargs)
        self._radius = radius  # major radius
        self._r = r  # minor radius

    @property
    def surface_area(self):
        return 4*np.pi*np.pi*self._radius*self._r

    @property
    def volume(self):
        return 2*np.pi*np.pi*self._radius*self._r*self._r

    def sdf(self, p):
        if len(p.shape) > 1:
            q = np.array([np.sqrt(p[:,0]**2 + p[:,2]**2)-self._radius,p[:,1]])
        else:
            q = np.array([np.sqrt(p[0]**2 + p[2]**2)-self._radius,p[1]])
        return np.linalg.norm(q)-self._r
    
class MeshShape(Shape):
    def __init__(self, mesh, **kwargs):
        super(MeshShape, self).__init__(**kwargs)
        self._surface_area = None
        self.__tree = None
        self._mesh = mesh
        self._set_radius()
        
    @property
    def surface_area(self):
        if self._surface_area is None:
            self._surface_area = np.sum(self._mesh._faces['area'][self._mesh._faces['halfedge']!=-1])
        return self._surface_area
    
    def _set_radius(self):
        diameter = 0
        for vi in self._mesh._vertices:
            if vi['halfedge'] == -1:
                continue
            for vj in self._mesh._vertices:
                if vj['halfedge'] == -1:
                    continue
                d = vi['position'] - vj['position']
                d2 = np.sum(d*d)
                if d2 > diameter:
                    diameter = d2
        self._radius = diameter/2.0
    
    @property
    def _tree(self):
        if self._tree is None:
            import scipy.spatial
            vertices = self._mesh._vertices['position'][self._mesh._vertices['position']!=-1]
            self.__tree = scipy.spatial.cKDTree(vertices)
        return self.__tree
    
    def sdf_triangle(self, p, face):
        dot = lambda v, w: (v*w).sum()
        dot2 = lambda v: (v*v).sum()
        
        def clamp(v, lo, hi):
            if v < lo:
                return lo
            if hi < v:
                return hi
            return v
        
        # Grab the triangle info from the mesh
        h0 = face['halfedge']
        # a = face['area']
        h1 = self._mesh._halfedges[h0]['next']
        h2 = self._mesh._halfedges[h1]['next']
        v0 = self._mesh._halfedges[h0]['vertex']
        v1 = self._mesh._halfedges[h1]['vertex']
        v2 = self._mesh._halfedges[h2]['vertex']
        p0 = self._mesh._vertices[v0]['position']
        p1 = self._mesh._vertices[v1]['position']
        p2 = self._mesh._vertices[v2]['position']
        
        # Calculate vector distances
        p1p0 = p1 - p0
        p2p1 = p2 - p1
        p0p2 = p0 - p2
        pp0 = p - p0
        pp1 = p - p1
        pp2 = p - p2
        n = np.cross(p1p0, p0p2)
        
        s1 = np.sign(dot(np.cross(p1p0,n),pp0))
        s2 = np.sign(dot(np.cross(p2p1,n),pp1))
        s3 = np.sign(dot(np.cross(p0p2,n),pp2))
        if (s1+s2+s3) < 2:
            f = np.min(np.min(
                      dot2(p1p0*clamp(dot(p2p1,pp0)/dot2(p1p0),0.0,1.0)-pp0),
                      dot2(p2p1*clamp(dot(p2p1,pp1)/dot2(p2p1),0.0,1.0)-pp1)),
                      dot2(p0p2*clamp(dot(p0p2,pp2)/dot2(p0p2),0.0,1.0)-pp2))
        else:
            f = dot(n,pp0)*dot(n,pp0)*dot2(n)
        
        return np.sqrt(f)
    
    def sdf(self, p):
        dist = []
        for _p in p:
            # Find the closest triangles in the mesh
            vertices = self._tree.query(_p, 20)
            faces = self._mesh._faces[self._mesh._halfedges['face'][self._mesh._vertices['halfedge'][vertices]]]

            # Evaluate sdf_triangles for each triangle
            d = 2*self._radius
            for face in faces:
                dt = self.sdf_triangle(_p, face)
                if dt < d:
                    d = dt
            dist.append(d)

        return np.array(dist)
import numpy as np
from ch_shrinkwrap import sdf, sdf_octree
from ch_shrinkwrap import util

class Shape:
    def __init__(self, **kwargs):
        """
        A general class for generating geometric shapes, and tools to compare 
        these shapes with externally-generated approximations. All Shapes are
        based in constructive solid geometry.
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
            ot = sdf_octree.cSDFOctree([-1.0*r, r, -1.0*r, r, -1.0*r, r], self.sdf, self._density, eps)
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
        return sdf.sphere(p, self._radius)

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
        return sdf.ellipsoid(p, self._a, self._b, self._c)

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
        return sdf.torus(p, self._radius, self._r)

class Tetrahedron(Shape):
    def __init__(self, v0, v1, v2, v3, **kwargs):
        super(Tetrahedron, self).__init__(**kwargs)
        d01 = util.dot2(v0-v1)
        d02 = util.dot2(v0-v2)
        d03 = util.dot2(v0-v3)
        d12 = util.dot2(v1-v2)
        d13 = util.dot2(v1-v3)
        d23 = util.dot2(v2-v3)
        self._radius = np.sqrt(np.max([d01,d02,d03,d12,d13,d23]))
        self._v0 = v0
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3

    @property
    def surface_area(self):
        v01 = self._v1 - self._v0
        v12 = self._v2 - self._v1
        v03 = self._v3 - self._v0
        v23 = self._v3 - self._v2

        # Calculate areas of each of the triangluar faces
        a021 = ((util.fast_3x3_cross(-v01, v12)**2).sum())**0.5
        a013 = ((util.fast_3x3_cross(v01, v03)**2).sum())**0.5
        a032 = ((util.fast_3x3_cross(-v23, -v03)**2).sum())**0.5
        a123 = ((util.fast_3x3_cross(v23, -v12)**2).sum())**0.5

        return a021+a013+a032+a123

    @property 
    def volume(self):
        v30 = self._v0 - self._v3
        v31 = self._v1 - self._v3
        v32 = self._v2 - self._v3
        return (1/6)*abs((v30*util.fast_3x3_cross(v31,v32)).sum())
    
    def sdf(self, p):
        return sdf.tetrahedron(p, self._v0, self._v1, self._v2, self._v3)

class Box(Shape):
    def __init__(self, lx=10, ly=None, lz=None, **kwargs):
        super(Box, self).__init__(**kwargs)
        self._lx = lx
        self._ly = ly
        self._lz = lz
        if not self._ly:
            self._ly = self._lx
        if not self._lz:
            self._lz = self._lx
        self._radius = np.sqrt(self._lx**2+self._ly**2+self._lz**2)
    
    @property
    def volume(self):
        return self._lx*self._ly*self._lz

    @property
    def surface_area(self):
        return 2.0*(self._lx*self._ly + self._lx*self._lz + self._ly*self._lz)

    def sdf(self, p):
        return sdf.box(p, self._lx, self._ly, self._lz)

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
        if self.__tree is None:
            import scipy.spatial
            vertices = self._mesh._vertices['position'][self._mesh._vertices['halfedge']!=-1]
            self.__tree = scipy.spatial.cKDTree(vertices)
        return self.__tree
    
    def sdf_triangle(self, p, face):
        
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
        
        return sdf.triangle(p, p0, p1, p2)
    
    def sdf(self, p):
        # Find the closest triangles in the mesh
        _, vertices = self._tree.query(p, 5)
        faces = self._mesh._faces[self._mesh._halfedges['face'][self._mesh._vertices['halfedge'][vertices]]]

        # Evaluate sdf_triangles for each triangle
        d = 2*self._radius
        for face in faces:
            dt = self.sdf_triangle(p, face)
            if np.abs(dt) < np.abs(d):
                d = dt

        return d

class UnionShape(Shape):
    def __init__(self, s0, s1, k=0, **kwargs):
        """
        Return the union of two shapes.

        Parameters
        ----------
            s0 : ch_shrinkwrap.shape.Shape
            s1 : ch_shrinkwrap.shape.Shape
            k : float
                Smoothing parameter
        """
        super(UnionShape, self).__init__(**kwargs)
        
        self._s0 = s0
        self._s1 = s1
        self._k = k

    def sdf(self, p):
        d0 = self._s0.sdf(p)
        d1 = self._s1.sdf(p)
        res = np.minimum(d0, d1)
        if self._k>0:
            h = np.maximum(self._k-np.abs(d0-d1),0.0)
            return res - h*h*0.25/self._k
        return res

class DifferenceShape(Shape):
    def __init__(self, s0, s1, k=0, **kwargs):
        """
        Return the difference of two shapes.

        Parameters
        ----------
            s0 : ch_shrinkwrap.shape.Shape
            s1 : ch_shrinkwrap.shape.Shape
            k : float
                Smoothing parameter
        """
        super(DifferenceShape, self).__init__(**kwargs)
        
        self._s0 = s0
        self._s1 = s1
        self._k = k

    def sdf(self, p):
        d0 = self._s0.sdf(p)
        d1 = self._s1.sdf(p)
        res = np.maximum(-d0, d1)
        if self._k>0:
            h = np.maximum(self._k-np.abs(-d0-d1),0.0)
            return res + h*h*0.25/self._k
        return res

class IntersectionShape(Shape):
    def __init__(self, s0, s1, k=0, **kwargs):
        """
        Return the intersection of two shapes.

        Parameters
        ----------
            s0 : ch_shrinkwrap.shape.Shape
            s1 : ch_shrinkwrap.shape.Shape
            k : float
                Smoothing parameter
        """
        super(IntersectionShape, self).__init__(**kwargs)
        
        self._s0 = s0
        self._s1 = s1
        self._k = k

    def sdf(self, p):
        d0 = self._s0.sdf(p)
        d1 = self._s1.sdf(p)
        res = np.maximum(d0,d1)
        if self._k>0:
            h = np.maximum(self._k-np.abs(d0-d1),0.0)
            return res + h*h*0.25/self._k
        return res
        
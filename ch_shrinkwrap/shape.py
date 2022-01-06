from ch_shrinkwrap import sdf, util

from PYME.simulation.locify import points_from_sdf

import math
import numpy as np

class Shape:
    def __init__(self, **kwargs):
        """
        A general class for generating geometric shapes, and tools to compare 
        these shapes with externally-generated approximations. All Shapes are
        based in constructive solid geometry.
        """
        self._density = None  # Shape density
        self._points = None  # Point queue representing Shape
        self._sigma = None
        self._normals = None  # normals for self._points
        self._radius = None  # maximum diameter of the shape/2
        self._sdf = None
        self.centroid = np.array([0,0,0],dtype=float)  # where is the shape centered?
        
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
        
    def __noise(self, model='poisson', **kw):
        """
        Noise model for 
        """

        self._sigma = util.loc_error(self._points.shape, model, **kw)
        return self._sigma*np.random.randn(*self._sigma.shape)
    
    def points(self, density=1, p=0.1, resample=False, noise='poisson', psf_width=250.0, mean_photon_count=300.0, return_normals=False):
        """
        Monte-Carlo sampling of uniform points on the Shape surface.
        
        Parameters
        ----------
            density : float
                Fluorophores per nm.
            p : float
                Likelihood that a fluorophore is detected.
            noise : str
                Noise model
            resample : bool
                Redo point sampling at each function call.
        """
        if resample or (self._points is None) or (self._density != density):
            self._density = density
            self._points = points_from_sdf(self.sdf, r_max=self._radius, centre=self.centroid, dx_min=(1.0/self._density)**(1.0/3.0), p=p).T
            if noise:
                self._points += self.__noise(noise, psf_width=psf_width, mean_photon_count=mean_photon_count)
            if return_normals:
                self._normals = sdf.sdf_normals(self._points.T, self.sdf).T

        if return_normals:
            return self._points, self._normals

        return self._points
    
    def surface_res(self, points):
        return np.sum(self.sdf(points)**2)

    def mse(self, points):
        return self.surface_res(points)/len(points)

class Sphere(Shape):
    def __init__(self, radius=2, **kwargs):
        Shape.__init__(self, **kwargs)
        self._radius = radius

    @property
    def surface_area(self):
        return 4*np.pi*self._radius*self._radius

    @property
    def volume(self):
        return (4.0/3.0)*np.pi*self._radius*self._radius*self._radius

    def sdf(self, p):
        return sdf.sphere(p, self._radius)

class Torus(Shape):
    def __init__(self, radius=2, r=0.05, **kwargs):
        Shape.__init__(self, **kwargs)
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
        Shape.__init__(self, **kwargs)
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

class Capsule(Shape):
    def __init__(self, start, end, radius=1, **kwargs):
        Shape.__init__(self, **kwargs)
        self._start = np.array(start,dtype=float)
        self._end = np.array(end,dtype=float)
        self._r = radius
        self._length = math.sqrt(util.dot2(self._end-self._start))
        self._radius = self._length/2.0 + radius
        self.centroid += 0.5*(self._start+self._end)
    
    @property
    def volume(self):
        return np.pi*self._r*self._r*((4.0/3.0)*self._r + self._length)
    
    @property
    def surface_area(self):
        return 2.0*np.pi*self._r*(2.0*self._r+self._length)

    def sdf(self, p):
        return sdf.capsule(p, self._start, self._end, self._r)

ThreeWayJunction = lambda h, r, centroid=[0,0,0], k=0: UnionShape(
                                    Capsule(centroid,centroid+[0,-h,0],r),
                                    UnionShape(
                                        Capsule(centroid, centroid+[-h/np.sqrt(2),h/np.sqrt(2),0],r),
                                        Capsule(centroid, centroid+[h/np.sqrt(2),h/np.sqrt(2),0],r), k
                                    ),
                                    k=0, centroid=centroid,
                                )
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
        Shape.__init__(self, **kwargs)
        
        self._s0 = s0
        self._s1 = s1
        self._k = k
        self._radius = max(self._s0._radius, self._s1._radius)

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
        Shape.__init__(self, **kwargs)
        
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
        
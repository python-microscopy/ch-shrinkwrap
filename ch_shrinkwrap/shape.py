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
        
    def __noise(self, model='exponential', **kw):
        """
        Noise model for 
        """

        self._sigma = util.loc_error(self._points.shape, model, **kw)
        return self._sigma*np.random.randn(*self._sigma.shape)
    
    def points(self, density=1, p=0.1, resample=False, noise='exponential', psf_width=280.0, mean_photon_count=600, 
               bg_photon_count=20, return_normals=False):
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
            self._points = points_from_sdf(self.sdf, r_max=self._radius, centre=self.centroid, 
                                           dx_min=(1.0/self._density)**(1.0/3.0), p=p).T
            if noise and psf_width is not None:
                self._points += self.__noise(noise, psf_width=psf_width, mean_photon_count=mean_photon_count, 
                                             bg_photon_count=bg_photon_count)
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
        return sdf.sphere(p-self.centroid[:,None], self._radius)

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
        return sdf.torus(p-self.centroid[:,None], self._radius, self._r)

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
    
class TaperedCapsule(Shape):
    def __init__(self, r1, r2, length=1, **kwargs):
        Shape.__init__(self, **kwargs)
        self._r1 = r1
        self._r2 = r2
        self._length = length
        self._radius = (length + max(r1, r2))/2.0
    
    def sdf(self, p):
        return sdf.tapered_capsule(p, self._r1, self._r2, self._length)
    
class RoundCone(Shape):
    def __init__(self, r1, r2, length=1, **kwargs):
        Shape.__init__(self, **kwargs)
        self._r1 = r1
        self._r2 = r2
        self._length = length
        self._radius = max(r1, r2, length)/2.0
    
    def sdf(self, p):
        return sdf.round_cone(p, self._r1, self._r2, self._length)

class Box(Shape):
    def __init__(self, halfwidth, r=0, **kwargs):
        Shape.__init__(self, **kwargs)
        self._r = r
        self._halfwidth = np.array(halfwidth)
        self._radius = np.max(halfwidth)

    @property
    def volume(self):
        return self._halfwidth*self._halfwidth*self._halfwidth
    
    @property
    def surface_area(self):
        return 2.0*np.sum(self._halfwidth**2)

    def sdf(self, p):
        return sdf.round_box(p-self.centroid[:,None], self._halfwidth, self._r)
    
class Sheet(Shape):
    def __init__(self, halfwidth, r=0, **kwargs):
        Shape.__init__(self, **kwargs)
        self._r = r
        self._halfwidth = np.array(halfwidth)
        self._radius = np.max(halfwidth)

    def sdf(self, p):
        return sdf.sheet(p-self.centroid[:,None], self._halfwidth, self._r)

def ThreeWayJunction(h, r, centroid=[0,0,0], k=0):
    centroid = np.array(centroid, dtype=float)
    return  UnionShape(
                Capsule(centroid,centroid+[0,-h,0],r),
                UnionShape(
                    Capsule(centroid, centroid+[-h/np.sqrt(2),h/np.sqrt(2),0],r),
                    Capsule(centroid, centroid+[h/np.sqrt(2),h/np.sqrt(2),0],r), k
                ),
                k=0, centroid=centroid,
            )

def ERSim(centroid=[0,0,0]):
    sheet_height = 100   # nm
    a, b = np.array([0,0,0]), np.array([400,-50,0])
    c, d = np.array([500,250,0]), np.array([0,217,0])
    e, f = np.array([0,-400,0]), np.array([-400,0,0])

    sheet0 = RotationShape(Box(np.array([66,83,sheet_height/4]), sheet_height/4), rz=np.pi/4)
    sheet1 = Box(np.array([50,50,sheet_height//4]), 1, centroid=np.array([0,133,0]))
    sheet2 = RotationShape(Box(np.array([33,33,sheet_height/4]), sheet_height/4), rz=7*np.pi/3, centroid=c)
    cap0 = Capsule(a,b,sheet_height//2)
    cap1 = Capsule(b,c,sheet_height//2)
    cap2 = Capsule(c,d,sheet_height//2)
    cap3 = Capsule(a,e,sheet_height//2)
    cap4 = Capsule(a,f,sheet_height//2)
    smooth = sheet_height//4
    struct = UnionShape(UnionShape(UnionShape(
                        UnionShape(sheet0,
                                UnionShape(cap0,
                                            UnionShape(cap1,
                                                        UnionShape(sheet2,cap2,k=smooth),
                                                        k=sheet_height),k=smooth), 
                                k=smooth), 
                        sheet1, k=smooth),cap3,k=smooth),cap4,k=smooth)
    return struct

def ERSim2(centroid=[0,0,0]):
    sheet_height = 100   # nm
    a, b = np.array([0,0,0]), np.array([400,-50,0])
    c, d = np.array([500,250,0]), np.array([0,217,0])
    e, f = np.array([0,-400,0]), np.array([-400,0,0])
    g, h = np.array([-40,0,-100]), np.array([-40,0,100])

    sheet0 = RotationShape(Sheet(np.array([126,100,sheet_height/3]), sheet_height/3), rz=np.pi/4)
    sheet1 = Sheet(np.array([50,50,sheet_height/3]), 1, centroid=np.array([0,133,0]))
    sheet2 = RotationShape(Sheet(np.array([33,33,sheet_height/3]), sheet_height/2), rz=7*np.pi/3, centroid=c)
    cap0 = Capsule(a,b,sheet_height//2)
    cap1 = Capsule(b,c,sheet_height//2)
    cap2 = Capsule(c,d,sheet_height//2)
    cap3 = Capsule(a,e,sheet_height//2)
    cap4 = Capsule(a,f,sheet_height//2)
    cap5 = Capsule(g,h,50)
    smooth = sheet_height/4
    struct = DifferenceShape(cap5, UnionShape(UnionShape(UnionShape(
                        UnionShape(sheet0,
                                UnionShape(cap0,
                                            UnionShape(cap1,
                                                        UnionShape(sheet2,cap2,k=smooth),
                                                        k=smooth),k=smooth),
                                k=smooth), 
                        sheet1, k=smooth),cap3,k=smooth),cap4,k=smooth),k=smooth)
    return struct

TwoToruses = lambda r, R: UnionShape(Torus(radius=R, r=r, centroid=np.array([-R,0,0])), Torus(radius=R, r=r, centroid=np.array([R,0,0])))

def NToruses(toruses, centroid=np.array([0,0,0])):
    """
    Generate a chain of N toruses. 

    Parameters
    ----------
    toruses: dict
        Dictionary of torus parameters. Key names do not matter.
        E.g. {'one': {'r': 30, 'R': 100}, 'two': {'r': 10, 'R': 75}, 'three': {'r': 30, 'R': 150}}
    centroid: np.array
        Centroid of first torus in the dictionary.
    """
    dt = toruses.pop(next(iter(toruses)))
    dcentroid = centroid.copy()
    if dcentroid[0] > 0:
        dcentroid[0] += float(dt['R'])  # TODO: Don't force along single axis?
    print(dt, dcentroid)
    
    torus = Torus(radius=float(dt['R']), r=float(dt['r']), centroid=dcentroid)
    if len(toruses) == 0:
        return torus
        
    return UnionShape(torus, NToruses(toruses, dcentroid + np.array([dt['R'], 0, 0])))

def DualCapsule(length, r, sep): 
    return UnionShape(Capsule(start=np.array([-sep/2,0,0]), end=np.array([-sep/2,length,0]), radius=r),
                      Capsule(start=np.array([sep/2,0,0]), end=np.array([sep/2,length,0]), radius=r))

class UnionShape(Shape):
    def __init__(self, s0, s1, k=0, **kwargs):
        """
        Return the union of two shapes.

        Parameters
        ----------
        s0 : shape.Shape
        s1 : shape.Shape
        k : float
            Smoothing parameter
        """
        Shape.__init__(self, **kwargs)
        
        self._s0 = s0
        self._s1 = s1
        self._k = k
        self._radius = self._s0._radius + self._s1._radius

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
        s0 : shape.Shape
        s1 : shape.Shape
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
        s0 : shape.Shape
        s1 : shape.Shape
        k : float
            Smoothing parameter
        """
        Shape.__init__(**kwargs)
        
        self._s0 = s0
        self._s1 = s1
        self._k = k
        self._radius = min(self._s0._radius, self._s1._radius)

    def sdf(self, p):
        d0 = self._s0.sdf(p)
        d1 = self._s1.sdf(p)
        res = np.maximum(d0,d1)
        if self._k>0:
            h = np.maximum(self._k-np.abs(d0-d1),0.0)
            return res + h*h*0.25/self._k
        return res

class RotationShape(Shape):
    def __init__(self, s0, rx=0.0, ry=0.0, rz=0.0, **kwargs):
        """
        Rotate a signed distance function.

        Parameters
        ----------
        s0 : shape.Shape
        rx: float
            Rotation in x-dir (rad)
        ry: float
            Rotation in y-dir (rad)
        rz: float
            Rotation in z-dir (rad)

        """
        Shape.__init__(self, **kwargs)

        self._s0 = s0

        sinx, cosx = np.sin(rx), np.cos(rx)
        siny, cosy = np.sin(ry), np.cos(ry)
        sinz, cosz = np.sin(rz), np.cos(rz)

        _rx = np.array([[1,0,0,],[0,cosx,-sinx],[0,sinx,cosx]])
        _ry = np.array([[cosy,0,siny],[0,1,0],[-siny,0,cosy]])
        _rz = np.array([[cosz,-sinz,0],[sinz,cosz,0],[0,0,1]])

        self._inv_r = np.linalg.inv(_rz @ (_ry @ _rx))

        self._radius = self._s0._radius

    def sdf(self, p):
        return self._s0.sdf(self._inv_r @ (p-self.centroid[:,None]))

class BentShape(Shape):
    """
    Bend a signed distance function.

    Parameters
    ----------
    s0 : shape.Shape
    k : float
        Bending amount
    """
    def __init__(self, s0, k=10.0):

        self._s0 = s0
        self._k = k
        self._radius = self._s0._radius

    def sdf(self, p):
        c = np.cos(self._k*p[0,:])
        s = np.sin(self._k*p[0,:])
        m = np.array([[c,-s],[s,c]])
        q = np.vstack([m*p[0,:], m*p[1,:], p[2,:]])
        return self._s0.sdf(q)

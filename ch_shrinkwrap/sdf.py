import math
import numpy as np
from . import util

def grad_sdf(pts, sdf, delta=0.1):
    """
    Gradient of the signed distance function, calculated via central differences.
    
    Parameters
    ----------
    pts : np.array
        3 x N array of points on which we evaluate the gradient
    sdf : function
        Signed-distance function. Expects 3xN vector of x, y, z coordinates.
    delta : float
        Shift in gradient direction.

    Returns
    -------
    np.array
        3 x N gradient for each point in pts
    """
    d2 = delta/2.0
    hx = np.array([d2,0,0])[:,None]
    hy = np.array([0,d2,0])[:,None]
    hz = np.array([0,0,d2])[:,None]
    dx = (sdf(pts + hx) - sdf(pts - hx))/delta
    dy = (sdf(pts + hy) - sdf(pts - hy))/delta
    dz = (sdf(pts + hz) - sdf(pts - hz))/delta
    
    return np.vstack([dx, dy, dz])

def sdf_normals(pts, sdf, delta=0.1):
    g = grad_sdf(pts, sdf, delta=delta)
    g_norm = np.linalg.norm(g, axis=0)
    return g/g_norm[None,:]

# Most SDFs from http://iquilezles.org/www/articles/distfunctions/distfunctions.htm.

def sphere(p, R):
    """
    p : np.array
        (3,N) point to calculate
    R : float
        Sphere radius
    """
    return np.linalg.norm(p, axis=0) - R

def torus(p, r, R):
    """
    p : np.array
        (3,N) point to calculate
    r : float
        inner radius
    R : float
        outer radius
    """
    q = np.array([np.sqrt(p[0,:]**2 + p[2,:]**2)-r,p[1,:]])
    return np.linalg.norm(q,axis=0)-R

def capsule(p, a, b, r):
    """Draw a capsule.

    Parameters
    ----------
    p : np.array
        (3,N) point to calculate
    a : np.array
        (3,) start of capsule
    b : np.array
        (3,) end of capsule
    r : float
        Capsule thickness
    """
    pa, ba = p - a[:,None], b - a
    h = np.clip( (pa*ba[:,None]).sum(0)/(ba*ba).sum(), 0.0, 1.0 )
    d = pa - ba[:,None]*h
    return np.sqrt((d*d).sum(0)) - r

def tetrahedron(p, v0, v1, v2, v3):
    """
    SDF of a tetrahedron, calculated as the intersection of the 
    planes formed by the triangles of the tetrahedron. Requires
    tetrahedron ordering as ordered_simps().
    
    Parameters
    ----------
        p : np.array
            (N, 3) array of xyz coordinates to search
        v0, v1, v2, v3 : np.array
            (3,) array of xyz coordinates forming a tetrahedron
    """
    p = np.atleast_2d(p)
    
    v01 = v1 - v0
    v12 = v2 - v1
    v03 = v3 - v0
    v23 = v3 - v2

    # Calculate normals of the tetrahedron
    # such that they point out of the tetrahedron
    n021 = util.fast_3x3_cross(-v01, v12)
    n013 = util.fast_3x3_cross(v01, v03)
    n032 = util.fast_3x3_cross(-v23, -v03)
    n123 = util.fast_3x3_cross(v23, -v12)

    # Define the planes
    # **(-0.5) is quite a bit faster than 1/sqrt
    # if you're willing to sacrifice some accuracy
    # (which we are)
    nn021 = n021*((util.fast_sum(n021*n021))**(-0.5))
    nn013 = n013*((util.fast_sum(n013*n013))**(-0.5))
    nn032 = n032*((util.fast_sum(n032*n032))**(-0.5))
    nn123 = n123*((util.fast_sum(n123*n123))**(-0.5))

    # Calculate the vectors from the point to the planes
    pv0 = p-v0
    p021 = (nn021*pv0).sum(1)
    p013 = (nn013*pv0).sum(1)
    p032 = (nn032*pv0).sum(1)
    p123 = (nn123*(p-v1)).sum(1)

    # Intersect the planes
    return np.max(np.vstack([p021, p013, p032, p123]).T,axis=1)

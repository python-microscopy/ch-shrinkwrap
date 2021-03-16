import math
import numpy as np
from ch_shrinkwrap import util

# Most SDFs from http://iquilezles.org/www/articles/distfunctions/distfunctions.htm.

def sphere(p, r):
    if len(p.shape) > 1:
        return np.sqrt(np.sum(p*p, axis=1)) - r
    return np.sqrt(np.sum(p*p)) - r

def box(p, lx, ly, lz):
    b = np.array([lx,ly,lz])
    if len(p.shape) > 1:
        q = np.abs(p) - b[None,:]
        r = np.linalg.norm(np.maximum(q,0.0)) + np.minimum(np.maximum(q[:,0],np.maximum(q[:,1],q[:,2])),0.0)
    else:
        q = np.abs(p) - b
        r = np.linalg.norm(np.maximum(q,0.0)) + np.minimum(np.maximum(q[0],np.maximum(q[1],q[2])),0.0)
    return r

def ellipsoid(p, a, b, c):
    r = np.array([a,b,c])
    pr = p/r
    prr = pr/r
    if len(p.shape) > 1:
        k0 = np.sqrt(np.sum(pr*pr,axis=1))
        k1 = np.sqrt(np.sum(prr*prr,axis=1))
        return k0*(k0-1.0)/k1
    k0 = np.sqrt(np.sum(pr*pr))
    k1 = np.sqrt(np.sum(prr*prr))
    return k0*(k0-1.0)/k1

def torus(p, r, R):
    if len(p.shape) > 1:
        q = np.array([np.sqrt(p[:,0]**2 + p[:,2]**2)-r,p[:,1]])
    else:
        q = np.array([np.sqrt(p[0]**2 + p[2]**2)-r,p[1]])
    return np.linalg.norm(q)-R

def capsule(p, a, b, r):
    """Draw a capsule.

    Parameters
    ----------
    p : np.array
        (3,) point to calculate
    a : np.array
        (3,) start of capsule
    b : np.array
        (3,) end of capsule
    r : float
        Capsule thickness
    """
    pa, ba = p - a, b - a
    h = util.clamp( util.dot(pa,ba)/util.dot2(ba), 0.0, 1.0 )
    return math.sqrt(util.dot2( pa - ba*h )) - r

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

def triangle(p, p0, p1, p2):
    # Calculate vector distances
    p1p0 = p1 - p0
    p2p1 = p2 - p1
    p0p2 = p0 - p2
    pp0 = p - p0
    pp1 = p - p1
    pp2 = p - p2
    n = util.fast_3x3_cross(p1p0, p0p2)
    
    s1 = np.sign(util.dot(util.fast_3x3_cross(p1p0,n),pp0))
    s2 = np.sign(util.dot(util.fast_3x3_cross(p2p1,n),pp1))
    s3 = np.sign(util.dot(util.fast_3x3_cross(p0p2,n),pp2))
    if (s1+s2+s3) < 2:
        f = np.minimum(np.minimum(
                    util.dot2(p1p0*util.clamp(util.dot(p2p1,pp0)/util.dot2(p1p0),0.0,1.0)-pp0),
                    util.dot2(p2p1*util.clamp(util.dot(p2p1,pp1)/util.dot2(p2p1),0.0,1.0)-pp1)),
                    util.dot2(p0p2*util.clamp(util.dot(p0p2,pp2)/util.dot2(p0p2),0.0,1.0)-pp2))
    else:
        f = util.dot(n,pp0)*util.dot(n,pp0)/util.dot2(n)
    
    return util.sign(util.dot(p,n))*np.sqrt(f)

def triangle_sdf2(p, pv):
    # Distances to multiple triangles
    p1p0 = pv[:,1] - pv[:,0]
    p2p1 = pv[:,2] - pv[:,1]
    p0p2 = pv[:,0] - pv[:,2]
    pp0 = p - pv[:,0]
    pp1 = p - pv[:,1]
    pp2 = p - pv[:,2]
    n = np.cross(p1p0, p0p2, axis=1)

    s1 = np.sign((np.cross(p1p0,n,axis=1)*pp0).sum(1))
    s2 = np.sign((np.cross(p2p1,n,axis=1)*pp1).sum(1))
    s3 = np.sign((np.cross(p0p2,n,axis=1)*pp2).sum(1))

    f = (n*pp0).sum(1)*(n*pp0).sum(1)/(n*n).sum(1)
    sign_mask = (s1+s2+s3) < 2
    f[sign_mask] = np.minimum(np.minimum(
                    ((p1p0*np.clip((p2p1*pp0).sum(1)/(p1p0*p1p0).sum(1),0.0,1.0)[:,None]-pp0)**2).sum(1),
                    ((p2p1*np.clip((p2p1*pp1).sum(1)/(p2p1*p2p1).sum(1),0.0,1.0)[:,None]-pp1)**2).sum(1)),
                    ((p0p2*np.clip((p0p2*pp2).sum(1)/(p0p2*p0p2).sum(1),0.0,1.0)[:,None]-pp2)**2).sum(1))[sign_mask]

    return np.sign((pp0*n).sum(1))*np.sqrt(f)

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
        Capsule radius
    """
    pa, ba = p - a[:,None], b - a
    h = np.clip( (pa*ba[:,None]).sum(0)/(ba*ba).sum(), 0.0, 1.0 )
    d = pa - ba[:,None]*h
    return np.sqrt((d*d).sum(0)) - r

def tapered_capsule(p, r1, r2, length):
    """
    Draw a tapered capsule.

    Parameters
    ----------
    p : np.array
        (3,N) point to calculate
    r1 : float
        Capsule minimum radius
    r2 : float
        Capsule maximum radius
    length : float
        Length of tapered capsule
    """

    x  = p[0,:]
    x1 = x/length
    
    r = np.sqrt((p[1:,:]**2).sum(0))
    
    rx = r1 + (r2-r1)*x1*x1
    
    p2 = p - np.array([1,0,0])[:,None]*length
    
    d = (x1 < 0)*(np.sqrt((p*p).sum(0))- r1) + \
        (x1>1)*(np.sqrt((p2*p2).sum(0))- r2) + \
        (x1 >= 0)*(x1 <=1)*(r-rx)

    return d

def tapered_ellipsoid(p, r1, r2, length):
    """
    Draw a tapered ellipsoid.

    Parameters
    ----------
    p : np.array
        (3,N) point to calculate
    r1 : float
        Capsule minimum radius
    r2 : float
        Capsule maximum radius
    length : float
        Length of tapered ellipsoid
    """

    x  = p[0,:]
    x1 = x/length
        
    rx = r1 + (r2-r1)*x1*x1
    
    p2 = p - np.array([1,0,0])[:,None]*length

    # bound the ellipsoid at the ends
    rr1 = np.array([r1, r1, r1/2])
    rr2 = np.array([r2, r2, r2/2])
    k0r1 = np.linalg.norm(p/rr1[:,None],axis=0)
    k1r1 = np.linalg.norm(p/(rr1**2)[:,None], axis=0)
    k0r2 = np.linalg.norm(p2/rr2[:,None], axis=0)
    k1r2 = np.linalg.norm(p2/(rr2**2)[:,None], axis=0)

    # use the exact formulation for the middle bit
    d = (x1<0)*k0r1*(k0r1-1.0)/k1r1 + \
        (x1>1)*k0r2*(k0r2-1.0)/k1r2 + \
        (x1 >= 0)*(x1 <=1)*ellipse(p[1:,:], rx, rx/2)  # k0rx*(k0rx-1.0)/k1rx

    return d

def ellipse(p, r1, r2):
    # vectorize this
    p = np.abs(p)
    ab = np.ones_like(p)
    ab[0,:] = r1
    ab[1,:] = r2

    # vectorized flip
    inds = p[0,:] > p[1,:]
    p[:,inds] = np.flip(p[:,inds], axis=0)
    ab[:,inds] = np.flip(ab[:,inds], axis=0)

    l = ab[1,:]*ab[1,:] - ab[0,:]*ab[0,:]
    m = ab[0,:]*p[0,:]/l
    m2 = m*m
    n = ab[1,:]*p[1,:]/l
    n2 = n*n
    c = (m2+n2-1.0)/3.0
    c3 = c*c*c
    q = c3 + m2*n2*2.0
    d = c3 + m2*n2
    g = m + m*n2

    h = 2.0*m*n*np.sqrt( d )
    s = np.sign(q+h)*np.abs(q+h)**(1.0/3.0)
    u = np.sign(q-h)*np.abs(q-h)**(1.0/3.0)
    rx = -s - u - c*4.0 + 2.0*m2
    ry = (s - u)*np.sqrt(3.0)
    rm = np.sqrt( rx*rx + ry*ry )
    co = (ry/np.sqrt(rm-rx)+2.0*g/rm-m)/2.0

    inds = d<0.0
    h[inds] = np.arccos(q[inds]/c3[inds])/3.0
    s[inds] = np.cos(h[inds])
    t = np.sin(h[inds])*np.sqrt(3.0)
    rx[inds] = np.sqrt( -c[inds]*(s[inds] + t + 2.0) + m2[inds] )
    ry[inds] = np.sqrt( -c[inds]*(s[inds] - t + 2.0) + m2[inds] )
    co[inds] = (ry[inds]+np.sign(l[inds])*rx[inds]+np.abs(g[inds])/(rx[inds]*ry[inds])- m[inds])/2.0
    
    r = ab * np.vstack([co, np.sqrt(1.0-co*co)])
    return np.linalg.norm(r-p,axis=0) * np.sign(p[1,:]-r[1,:])

def round_cone(p, r1, r2, length):

    b = (r1-r2)/length
    a = np.sqrt(1.0-b*b)

    q = np.vstack([np.sqrt(p[0,:]**2+p[2,:]**2), p[1,:]])
    k = (q*np.array([-b,a])[:,None]).sum(0)

    d = (q*np.array([a,b])[:,None]).sum(0) - r1
    d[k<0.0] = np.linalg.norm(q, axis=0)[k<0.0] - r1
    d[k>(a*length)] = np.linalg.norm((q-np.array([0.0,length])[:,None]), axis=0)[k>(a*length)] - r2

    return d

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

def round_box(p, w, r):
    """Draw a box of width w rounded by radius r.

    Parameters
    ----------
    p : np.array
        (3,N) point to calculate
    w : np.array
        (3,) halfwidth of box in x, y, z
    r : float
        Radius of rounded corners

    Returns
    -------
    np.array
        (N,) array of distances to box
    """
    w = np.array(w)
    q = np.abs(p) - w[:,None]
    return np.linalg.norm(np.maximum(q,0.0),axis=0) + np.minimum(np.maximum(q[0,:],np.maximum(q[1,:],q[2,:])),0.0) - r

def sheet(p, w, r):
    """Apply a dumbbell shape to a box.

    Parameters
    ----------
    p : np.array
        (3,N) point to calculate
    w : np.array
        (3,) halfwidth of box in x, y, z
    r : float
        Radius of rounded corners

    Returns
    -------
    np.array
        (N,) array of distances to sheet
    """
    w = np.array(w)

    q = np.abs(p) - w[:,None]
    m = np.maximum(q[0,:],np.maximum(q[1,:],q[2,:]))
    return np.minimum(np.linalg.norm(np.vstack([np.maximum(q[0,:],q[1,:])+r, q[2,:]+w[2]]),axis=0) - r, m)

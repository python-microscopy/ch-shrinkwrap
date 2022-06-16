from keyword import kwlist
import numpy as np

def fast_3x3_cross(a,b):
    # Quite a bit faster than np.cross
    # for 3x1 vectors
    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]
    
    vec = np.array([x,y,z])
    return vec

def fast_sum(vec):
    # Technically faster than numpy sum() operation for length 3 vectors
    # A wholly unnecessary optimization
    return vec[0]+vec[1]+vec[2]

def dot(v, w):
    return (v*w).sum()

def dot2(v):
    return (v*v).sum()

def clamp(v, lo, hi):
    if v < lo:
        return lo
    if hi < v:
        return hi
    return v

def sign(x):
    if x > 0:
        return 1
    return -1

def loc_error(shape, model=None, **kw):
    if model == 'exponential':
        l = np.vstack([np.random.exponential(kw['mean_photon_count'],10*shape[0]) for i in range(shape[1])]).T
        if type(kw['psf_width']) == float:
            sigma = np.vstack([(kw['psf_width']/2.355)/np.sqrt(l[:,i][l[:,i] > kw['bg_photon_count']][:shape[0]]) for i in range(shape[1])]).T
        else:
            sigma = np.vstack([(kw['psf_width'][i]/2.355)/np.sqrt(l[:,i][l[:,i] > kw['bg_photon_count']][:shape[0]]) for i in range(shape[1])]).T
    else:
        sigma = 10.0*np.ones(shape)

    return sigma

def point_inside_triangle(pt, tri):
    """
    Checks if a ((x,y,z),) point is in an ((v0,v1,v2), (x, y, z)) triangle
    using barycentric coordinates.
    """
    v0 = tri[2,:]-tri[0,:]
    v1 = tri[1,:]-tri[0,:]
    v2 = pt-tri[0,:]
    
    dot00 = (v0*v0).sum()
    dot01 = (v0*v1).sum()
    dot02 = (v0*v2).sum()
    dot11 = (v1*v1).sum()
    dot12 = (v1*v2).sum()
    
    inv_denom = 1 / (dot00*dot11 - dot01*dot01)
    u = (dot11 * dot02 - dot01*dot12) * inv_denom
    v = (dot00 * dot12 - dot01*dot02) * inv_denom
    
    return (u >= 0) and (v >= 0) and (u + v < 1)
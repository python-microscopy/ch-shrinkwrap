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

def noise(shape, model=None, **kw):
    if model == 'poisson':
        if kw['psf_width'] > 0:
            l = np.vstack([np.random.poisson(kw['mean_photon_count'],10*shape[0]) for i in range(shape[1])]).T
            # note: currently isotropic
            sigma = np.vstack([kw['psf_width']/np.sqrt(l[:,i][l[:,i] > kw['mean_photon_count']][:shape[0]]) for i in range(shape[1])]).T
    else:
        sigma = 10.0*np.ones(shape)

    return sigma*np.random.randn(*sigma.shape)

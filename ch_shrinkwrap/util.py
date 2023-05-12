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

def surf_residuals(surf, points, sigma):
    from PYME.experimental import isosurface
    import matplotlib.pyplot as plt
    from scipy import stats
        
    d = isosurface.distance_to_mesh(points, surf, smooth=False)

    f = plt.figure()
    a1, a2 = f.subplots(2, 1)
    a1.hist(d, np.linspace(-100, 100, 500))
    a1.grid()
    a1.set_xlabel('Distance from surface [nm]')
    a1.set_ylabel('Frequency')
    a1.set_title('Surface residuals')

    #a = plt.subplot()
    a2.hist(np.abs(d), np.linspace(0, 100, 500), density=True, label='Experimental')

    me = np.median(sigma)
    print(me)
    x = np.linspace(0, 100, 1000)
    a2.plot(x, stats.chi(3).pdf(x/me)/(me), label='Predicted')

    a2.grid()
    #a2.set_xlabel('Squared distance [nm^2]')
    a2.set_xlabel('Absolute distance from surface [nm]')
    a2.set_ylabel('Frequency')
    a2.legend()



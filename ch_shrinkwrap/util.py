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
    a1 = f.subplots(1, 1)
    a1.hist(d, np.linspace(-100, 100, 500), density=True)
    a1.grid()
    a1.set_xlabel('Distance from surface [nm]')
    a1.set_ylabel('Frequency')
    a1.set_title('Surface residuals')

    ##a = plt.subplot()
    #a2.hist(np.abs(d), np.linspace(0, 100, 500), density=True, label='Experimental')

    me = np.median(sigma)
    #print(me)
    x = np.linspace(-100, 100, 1000)
    a1.plot(x, 0.5*stats.chi(3).pdf(np.abs(x)/me)/(me), label='Predicted')

    #a2.grid()
    #a2.set_xlabel('Squared distance [nm^2]')
    #a2.set_xlabel('Absolute distance from surface [nm]')
    #a2.set_ylabel('Frequency')
    a1.legend()

    # f = plt.figure()
    # a1, a2 = f.subplots(2, 1)
    # c, _, _ = a2.hist(np.abs(d), np.linspace(0, 100, 500), density=True, label='Experimental')
    # print('c.sum()', c.sum())
    
    # b = np.arange(0, 30, 1)
    # c, _ = np.histogram(sigma.ravel(), b, density=True)
    # bm = 0.5*(b[:-1] + b[1:])

    # d_ = 0
    # #print(b, c, bm, sigma)
    # for c_, b_ in zip(c, bm):
    #     for r in np.arange(0.09, 1.01, .1):
    #         p = 0.1*(3/2)*(1.0-r*r)
    #         d_ += p*c_*stats.chi(3).pdf(x/(b_*r))/(b_*r)

    # print('d_.sum()', d_.sum())
    # a2.plot(x, 0.5*d_, label='Predicted')

    # # d_ = 0
    # # #print(b, c, bm, sigma)
    # # #for c_, b_ in zip(c, bm):
    # # for r in np.arange(0.09, 1.01, .1):
    # #     p = 0.1*(3/2)*(1.0-r*r)
    # #     d_ += p*stats.chi(3).pdf(x/(me*r))/(me*r)

    # # print('d_.sum()', d_.sum())
    # # a2.plot(x, 0.5*d_, label='Predicted')

    # a2.grid()
    # #a2.set_xlabel('Squared distance [nm^2]')
    # a2.set_xlabel('Absolute distance from surface [nm]')
    # a2.set_ylabel('Frequency')
    # a2.legend()

    






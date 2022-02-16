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
    if model == 'poisson':
        if kw['psf_width'][0] > 0:
            l = np.vstack([np.random.poisson(kw['mean_photon_count'],10*shape[0]) for i in range(shape[1])]).T
            sigma = np.vstack([kw['psf_width'][i]/np.sqrt(l[:,i][l[:,i] > kw['mean_photon_count']][:shape[0]]) for i in range(shape[1])]).T
    else:
        sigma = 10.0*np.ones(shape)

    return sigma

def generate_smlm_pointcloud_from_shape(shape, density=1, p=0.0001, psf_width=250.0, 
                                        mean_photon_count=300,
                                        noise_fraction=0.1, save_fn=None, **kw):
    """
    Generate an SMLM point cloud from a Shape object. 
    
    Parameters
    ----------
    density : float
        Fluorophores per nm.
    p : float
        Likelihood that a fluorophore is detected.
    psf_width : float
        Width of the microscope point spread function
    mean_photon_count : float
        Average number of photons within a PSF
    noise_fraction : float
        Fraction of total points that will be noise
    save_fn : str
        Complete path and name of the .txt file describing
        where to save this simulation 
    """

    # simulate the points
    cap_points = shape.points(density=density, p=p, psf_width=psf_width, 
                            mean_photon_count=mean_photon_count, 
                            resample=True)
    # find the precision of each simulated point
    cap_sigma = shape._sigma
    
    # simualte clusters at each of the points
    cap_points, cap_sigma = smlmify_points(cap_points, cap_sigma, psf_width=psf_width, 
                                           mean_photon_count=mean_photon_count, 
                                           **kw)

    # set up bounding box of simulation to decide where to put background
    no, scale = noise_fraction, 1.2
    bbox = [np.min(cap_points[:,0]), np.min(cap_points[:,1]), 
            np.min(cap_points[:,2]), np.max(cap_points[:,0]),
            np.max(cap_points[:,1]), np.max(cap_points[:,2])]
    bbox = [scale*x for x in bbox]
    xl, yl, zl, xu, yu, zu = bbox
    xn, yn, zn = xu-xl, yu-yl, zu-zl
    ln = int(no*len(cap_points)/(1.0-no))

    # simulate background points random uniform over the bounding box
    noise_points = np.random.rand(ln,3)*(np.array([xn,yn,zn])[None,:]) \
                   + (np.array([xl,yl,zl])[None,:])
    noise_sigma = loc_error(noise_points.shape, model='poisson', 
                            psf_width=psf_width, 
                            mean_photon_count=mean_photon_count)
    
    # simulate clusters at each of the random noise points
    noise_points, noise_sigma = smlmify_points(noise_points, noise_sigma, psf_width=psf_width, 
                                               mean_photon_count=mean_photon_count)
    
    # stack the regular and noise points
    points = np.vstack([cap_points,noise_points])
    sigma = np.vstack([cap_sigma,noise_sigma])
    s = np.sqrt((sigma*sigma).sum(1))

    # pass metadata associated with this simulation
    md = {'shape': shape.__str__(), 'density': density, 'p': p, 'psf_width': psf_width, 
          'mean_photon_count': mean_photon_count, 'noise_fraction': noise_fraction}
    
    if save_fn is not None:
        import os
        _, ext = os.path.splitext(save_fn)
        if ext == '.txt':
            np.savetxt(save_fn, np.vstack([points.T,s]).T, header="x y z sigma")
        elif ext == '.hdf':
            from PYME.IO.tabular import ColumnSource
            ds = ColumnSource(x=points[:,0], y=points[:,1], z=points[:,2], sigma=s, sigma_x=sigma[:,0], sigma_y=sigma[:,1], sigma_z=sigma[:,2])
            ds.to_hdf(save_fn)
        else:
            raise UserWarning('File type unrecognized. File was not saved.')
        md['filename'] = save_fn
        
    return md

def smlmify_points(points, sigma, psf_width=250.0, mean_photon_count=300.0, max_points_per_cluster=10, max_points=None):
    # simulate clusters of points around each noise point
    noise_points = np.vstack([np.random.normal(points, sigma) for i in range(max_points_per_cluster)])
    
    sz = points.shape[0] if max_points is None else max_points
    
    # extract only sz points, some of the originals may disappear
    noise_points = noise_points[np.random.choice(np.arange(noise_points.shape[0]), size=sz, replace=False)]
    
    # Generate new sigma for each of these points
    noise_sigma = loc_error(noise_points.shape, model='poisson', 
                            psf_width=psf_width, 
                            mean_photon_count=mean_photon_count)
    
    return noise_points, noise_sigma

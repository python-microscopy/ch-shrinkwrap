"""
Tools for comparing mesh data sets to theoertical object they are
approximating. Motivation for ordered pairs, Hausdorff, mean distance 
and smoothness functions comes from 

Berger, Matt, Josh Levine, Luis Gustavo Nonato, Gabriel Taubin, and 
Claudio T. Silva. "An End-to-End Framework for Evaluating Surface 
Reconstruction." Technical. SCI Institute, University of Utah, 2011. 
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.190.6244.

"""

from ch_shrinkwrap import _membrane_mesh as membrane_mesh
from ch_shrinkwrap import util
from ch_shrinkwrap import shape

from typing import Tuple, Union, Optional

import numpy.typing as npt

import os
import sys
import itertools
import scipy.spatial
import numpy as np

import logging
logger = logging.getLogger(__name__)

from ch_shrinkwrap.sdf import sdf_normals

if sys.platform == 'darwin':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def points_from_mesh(mesh, dx_min: float = 5, p: float = 1.0,
                     return_normals: bool = False) -> Union[npt.ArrayLike, 
                                                            Tuple[npt.ArrayLike, npt.ArrayLike]]:
    """
    Generate uniform sampling of points on a mesh.

    Parameters
    ----------
    mesh : ch_shrinkwrap._membrane_mesh.MembraneMesh or PYME.experimental._triangle_mesh.TriangleMesh
        Mesh representation of an object.
    dx_min : float
        The target side length of a voxel. Sets maximum sampling rate.
    p : float
        Monte-Carlo acceptance probability.

    Returns
    -------
    """

    # find triangles and normals
    tris = mesh._vertices['position'][mesh.faces]  # (N_faces, (v0,v1,v2), (x,y,z))
    #norms = mesh._faces['normal'][mesh._faces['halfedge'] != -1]  # (N_faces, (x,y,z))
    norms = np.cross((tris[:,2,:]-tris[:,1,:]),(tris[:,0,:]-tris[:,1,:]))  # (N_faces, (x,y,z))
    nn = np.linalg.norm(norms,axis=1)

    # If a triangle is degenerate (has parallel sides) and zero area we get a 0 length cross-product 
    # (and a whole pile of NaNs once we divide by nn)
    # calculate a mask to work out where this happens, and exclude the zero area triangles
    nan_mask = nn != 0 
    #logger.debug(f'nn.shape: {nn.shape}, tris.shape: {tris.shape}')
    if np.any(~nan_mask):
        logger.warning('Detected zero error triangles in mesh')

    #logger.debug(f'tris[:5,:,:]: {tris[:5,:,:]}' )
    
    norms = norms[nan_mask]/nn[nan_mask,None]
    tris = tris[nan_mask,:,:]
    #logger.debug(f'norms.shape: {norms.shape}, tris.shape: {tris.shape}' )
    #logger.debug(f'tris[:5,:,:]: {tris[:5,:,:]}' )

    # construct orthogonal vectors to form basis of triangle plane
    v0 = tris[:,1,:]-tris[:,0,:]     # (N_faces, (x,y,z))
    e0n = np.linalg.norm(v0,axis=1)  # (N_faces,)
    e0 = v0/e0n[:,None]              # (N_faces, (x,y,z))
    e1 = np.cross(norms,e0,axis=1)   # (N_faces, (x,y,z))

    # Decompose triangle positions into multiples of e0 and e1
    x0 = (tris[:,0,:]*e0).sum(1)              # (N_faces,)
    y0 = (tris[:,0,:]*e1).sum(1)
    x1 = (tris[:,1,:]*e0).sum(1) 
    y1 = (tris[:,1,:]*e1).sum(1) 
    x2 = (tris[:,2,:]*e0).sum(1) 
    y2 = (tris[:,2,:]*e1).sum(1)

    # Compute bounds of grid for each triangle
    x0x1x2 = np.vstack([x0,x1,x2]).T  # (N_faces, 3)
    y0y1y2 = np.vstack([y0,y1,y2]).T
    xl = np.min(x0x1x2, axis=1)       # (N_faces,)
    xu = np.max(x0x1x2, axis=1)
    yl = np.min(y0y1y2, axis=1)
    yu = np.max(y0y1y2, axis=1)

    # Compute slopes of lines for each triangle
    x1x0 = x1-x0
    x2x1 = x2-x1
    x0x2 = x0-x2
    m0 = (y1-y0)/x1x0
    m0[x1x0 == 0] = 0
    m1 = (y2-y1)/x2x1
    m1[x2x1 == 0] = 0
    m2 = (y0-y2)/x0x2
    m2[x0x2 == 0] = 0
    s1 = np.sign(m1)
    s2 = np.sign(m2)

    d = []

    

    # for each triangle...
    for i in range(tris.shape[0]):
        

        # create a grid of points for this triangle
        x = np.arange(xl[i]-x0[i]-dx_min/2, xu[i]-x0[i], dx_min)  # normalize coordinates to vertex 0
        y = np.arange(yl[i]-y0[i]-dx_min/2, yu[i]-y0[i], dx_min)
        X, Y = np.meshgrid(x,y)

        # Mask points inside the triangle by thresholding on everything above the edge from 0 to 1 (e0)
        # and everything inside the lines going from x1 to x2 and x2 to x0 (consistent winding)
        X_mask = (Y > X*m0[i]) & (s1[i]*Y > s1[i]*(y1[i]-y0[i] + (X-x1[i]+x0[i])*m1[i])) & (s2[i]*Y < s2[i]*(y2[i]-y0[i] + (X-x2[i]+x0[i])*m2[i]))

        # return the masked points
        pos = X[X_mask].ravel()[:,None]*e0[i,None,:] + Y[X_mask].ravel()[:,None]*e1[i,None,:] + tris[i,0,:]

        d.append(pos)

    d = np.vstack(d)

    subsamp = np.random.choice(np.arange(d.shape[0]), size=int(p*d.shape[0]), replace=False)

    if return_normals:
        # TODO: This repeats a fair bit of distance_to_mesh

        # Create a list of face centroids for search
        face_centers = mesh._vertices['position'][mesh.faces].mean(1)

        # Construct a kdtree over the face centers
        tree = scipy.spatial.cKDTree(face_centers)

        _, _faces = tree.query(d, k=1)

        normals = mesh._faces['normal'][mesh._faces['halfedge'] != -1][_faces]

        return d[subsamp], normals[subsamp]

    return d[subsamp]

def average_squared_distance(points0 : npt.ArrayLike, points1: npt.ArrayLike) -> Tuple[float, float]:
    """
    Return the average squared distances between nearest neighbor correspondences of two point clouds.

    Parameters
    ----------
    points0 : np.array
        Point cloud
    points1 : np.array
        Point cloud

    Returns
    -------
    points0 : float
        Average squared distance of points1 from points0
    points1 : float 
        Average squared distance of points0 from points1
    """
    points0_tree = scipy.spatial.cKDTree(points0)
    points1_tree = scipy.spatial.cKDTree(points1)

    points0_err, _ = points0_tree.query(points1, k=1)
    points1_err, _ = points1_tree.query(points0, k=1)

    points0_mse = np.nansum(points0_err**2)/len(points0_err)
    points1_mse = np.nansum(points1_err**2)/len(points1_err)

    return points0_mse, points1_mse

def generate_smlm_pointcloud_from_shape(shape_name : str, shape_params : dict, density : float = 1, 
                                        p : float = 0.0001, psf_width : Union[float, Tuple, None] = 250.0, 
                                        mean_photon_count : int = 300, bg_photon_count : int = 20.0,
                                        noise_fraction : float = 0.1) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """
    Generate an SMLM point cloud from a Shape object. 
    
    Parameters
    ----------
    shape_name : str
        Name of shape
    shape_params : dict
        Arguments to feed to shape
    density : float
        Fluorophores per nm.
    p : float
        Likelihood that a fluorophore is detected.
    psf_width : float or tuble
        Width of the microscope point spread function along (x,y,z)
    mean_photon_count : int
        Average number of photons within a PSF
    bg_photon_count : int
        Average number of photons in an empty pixel.
    noise_fraction : float
        Fraction of total points that will be noise
    """


    test_shape = getattr(shape, shape_name)(**shape_params)

    # simulate the points
    cap_points = test_shape.points(density=density, p=p, psf_width=psf_width, 
                            mean_photon_count=mean_photon_count, 
                            bg_photon_count=bg_photon_count,
                            resample=True)

    # find the precision of each simulated point
    cap_sigma = test_shape._sigma

    if psf_width is None:
        normals = sdf_normals(cap_points.T, test_shape.sdf).T
        return cap_points, normals, cap_sigma
    
    # simulate clusters at each of the points
    cap_points, cap_sigma = smlmify_points(cap_points, cap_sigma, psf_width=psf_width, 
                                                     mean_photon_count=mean_photon_count, 
                                                     bg_photon_count=bg_photon_count)

    if noise_fraction > 0:
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
        noise_sigma = util.loc_error(noise_points.shape, model='exponential', 
                                    psf_width=psf_width, 
                                    mean_photon_count=mean_photon_count,
                                    bg_photon_count=bg_photon_count)
        
        # simulate clusters at each of the random noise points
        noised_points, noised_sigma = smlmify_points(noise_points, noise_sigma, psf_width=psf_width, 
                                                     mean_photon_count=mean_photon_count,
                                                     bg_photon_count=bg_photon_count)
        
        # stack the regular and noise points
        points = np.vstack([cap_points,noised_points])
        sigma = np.vstack([cap_sigma,noised_sigma])
    else:
        points = cap_points
        sigma = cap_sigma

    normals = sdf_normals(points.T, test_shape.sdf).T

    return points, normals, sigma

def smlmify_points(points : npt.ArrayLike, sigma : npt.ArrayLike, psf_width : Union[float, Tuple, None] = 250.0, 
                   mean_photon_count : int = 300, bg_photon_count : int = 20, max_points_per_cluster : int = 10, 
                   max_points : Optional[int] = None) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    # simulate clusters of points around each noise point
    noise_points = np.vstack([np.random.normal(points, sigma) for i in range(max_points_per_cluster)])
    
    sz = points.shape[0] if max_points is None else max_points
    
    # extract only sz points, some of the originals may disappear
    noise_points = noise_points[np.random.choice(np.arange(noise_points.shape[0]), size=sz, replace=False)]
    
    # Generate new sigma for each of these points
    noise_sigma = util.loc_error(noise_points.shape, model='exponential', 
                                 psf_width=psf_width, 
                                 mean_photon_count=mean_photon_count,
                                 bg_photon_count=bg_photon_count)
    
    return noise_points, noise_sigma

def testing_parameters(test_d : dict) -> Tuple[dict, dict]:
    """Expand YAML dict to flat list. This YAML dict is defined in README.md and there is
    an example at ch_shrinkwrap/test_example.yaml. """

    # System
    psf_widths = list(itertools.product(test_d['system']['psf_width_x'],
                                   test_d['system']['psf_width_y'],
                                   test_d['system']['psf_width_z']))
    mean_photon_counts = test_d['system']['mean_photon_count']
    bg_photon_counts = test_d['system']['bg_photon_count']

    # Shape
    shape_type = test_d['shape']['type']
    shape_params = test_d['shape']['parameters']

    # Point cloud
    cloud_densities = test_d['point_cloud']['density']
    cloud_p = test_d['point_cloud']['p']
    cloud_noise_fraction = test_d['point_cloud']['noise_fraction']

    # Dual marching cubes
    march_density = test_d['dual_marching_cubes']['threshold_density']
    march_points = test_d['dual_marching_cubes']['n_points_min']

    densities = list(zip(cloud_densities, cloud_p, march_density, march_points))

    # Shrinkwrapping
    sw_iters = test_d['shrinkwrapping']['max_iters']
    sw_curv = test_d['shrinkwrapping']['curvature_weight']
    sw_remesh = test_d['shrinkwrapping']['remesh_frequency']
    sw_punch = test_d['shrinkwrapping']['punch_frequency']
    sw_hole_rad = test_d['shrinkwrapping']['min_hole_radius']
    sw_neck_iter = test_d['shrinkwrapping']['neck_first_iter']
    sw_neck_low = test_d['shrinkwrapping']['neck_threshold_low']
    sw_neck_high = test_d['shrinkwrapping']['neck_threshold_high']

    # SPR
    spr_spn = test_d['screened_poisson']['samplespernode']
    spr_weight = test_d['screened_poisson']['pointweight']
    spr_iters = test_d['screened_poisson']['iters']
    spr_k = test_d['screened_poisson']['k']

    # common parameters
    param_list = [psf_widths, mean_photon_counts, bg_photon_counts,
                  shape_type, shape_params, densities, 
                  cloud_noise_fraction]

    # shrinkwrapping-specific parameters
    sw_param_list = param_list + [sw_iters, sw_curv, sw_remesh,
                     sw_punch, sw_hole_rad, sw_neck_iter,
                     sw_neck_low, sw_neck_high]

    # spr-specific parameters
    spr_param_list = param_list + [spr_spn, spr_weight, spr_iters, spr_k]

    sw_list = itertools.product(*sw_param_list)
    spr_list = itertools.product(*spr_param_list)

    # Re-cast to dictionary 
    param_keys = ['psf_width', 'mean_photon_count', 'bg_photon_count',
                  'shape_name', 'shape_params', 'density',
                  'p', 'threshold_density', 'n_points_min',
                  'noise_fraction']

    sw_keys = param_keys + ['max_iter', 'curvature_weight', 'remesh_frequency',
                     'punch_frequency', 'min_hole_radius', 'neck_first_iter',
                     'neck_threshold_low', 'neck_threshold_high']

    spr_keys = param_keys + ['samplespernode', 'pointweight', 'iters', 'k']

    def to_dict(sw_list, sw_keys):
        sw_dicts = []
        for k, it in enumerate(sw_list):
            sw_dicts.append({})
            i = 0
            for el in it: 
                if i == 5:
                    # densities
                    for j in range(4):
                        sw_dicts[k][sw_keys[i]] = el[j]
                        i += 1
                else:
                    sw_dicts[k][sw_keys[i]] = el
                    i += 1
        return sw_dicts

    sw_dicts = to_dict(sw_list, sw_keys)
    spr_dicts = to_dict(spr_list, spr_keys)

    return sw_dicts, spr_dicts

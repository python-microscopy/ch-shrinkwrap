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

from typing import Optional, Tuple, Union

import numpy.typing as npt

import os
import sys
import time
import scipy.spatial
import numpy as np

import itertools
import yaml
import time
from functools import partial
import uuid

from tables.exceptions import HDF5ExtError

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
    norms = norms/nn[:,None]

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
                                        p : float = 0.0001, psf_width : Union[float, Tuple] = 250.0, 
                                        mean_photon_count : float = 300, bg_photon_count : float = 20.0,
                                        noise_fraction : float = 0.1) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
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
    mean_photon_count : float
        Average number of photons within a PSF
    noise_fraction : float
        Fraction of total points that will be noise
    save_fn : str
        Complete path and name of the .txt file describing
        where to save this simulation 
    """


    test_shape = getattr(shape, shape_name)(**shape_params)

    # simulate the points
    cap_points = test_shape.points(density=density, p=p, psf_width=psf_width, 
                            mean_photon_count=mean_photon_count, 
                            bg_photon_count=bg_photon_count,
                            resample=True)
    # find the precision of each simulated point
    cap_sigma = test_shape._sigma
    
    # simualte clusters at each of the points
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
        noise_points, noise_sigma = smlmify_points(noise_points, noise_sigma, psf_width=psf_width, 
                                                mean_photon_count=mean_photon_count,
                                                bg_photon_count=bg_photon_count)
        
        # stack the regular and noise points
        points = np.vstack([cap_points,noise_points])
        sigma = np.vstack([cap_sigma,noise_sigma])
    else:
        points = cap_points
        sigma = cap_sigma

    return points, sigma

def smlmify_points(points, sigma, psf_width=250.0, mean_photon_count=300.0, bg_photon_count=20.0,
                   max_points_per_cluster=10, max_points=None):
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

def compute_mesh_metrics(yaml_file, test_shape, dx_min=1, p=1.0, psf_width=250.0, 
                         mean_photon_count=300.0, bg_photon_count=20.0):
    """
    yaml_file: fn
        File containing list of meshes
    shape : ch_shrinkwrap.shape.Shape
        Theoeretical shape created from a signed distance function
    dx_min : float
        The target side length of a voxel. Sets sampling rate.
    p : float
        Monte-Carlo acceptance probability.
    """

    d, new_d = [], []
    with open(yaml_file) as f:
        d = yaml.safe_load(f)
    
    test_points, test_normals = test_shape.points(density=1.0/(dx_min**3), p=p, 
                                           psf_width=psf_width, 
                                           mean_photon_count=mean_photon_count, 
                                           bg_photon_count=bg_photon_count,
                                           resample=True, noise=None, 
                                           return_normals=True)
    failed = 0
    for el in d:
        mesh_d = el.get('mesh')
        if mesh_d is not None:
            print(mesh_d['filename'])
            
            try:
                # load mesh
                mesh = membrane_mesh.MembraneMesh.from_stl(mesh_d['filename'])

                # calculate mean squared error
                # vecs = mesh._vertices[mesh.faces]['position']
                # errors = test_shape.sdf(vecs.mean(1).T)
                # mse = np.nansum(errors**2)/len(errors)

                # Calculate distance and angle stats
                # hd, md, ha, ma = test_points_mesh_stats(test_points, 
                #                                         test_normals, 
                #                                         mesh,
                #                                         dx_min=dx_min,
                #                                         p=p)
                test_mse, mesh_mse = test_points_mesh_stats(test_points, 
                                                            test_normals, 
                                                            mesh,
                                                            dx_min=dx_min,
                                                            p=p, hausdorff=False)
                
                # mesh_d['mse'] = float(mse)
                # mesh_d['hausdorff_distance'] = float(hd)
                # mesh_d['mean_distance'] = float(md)
                # mesh_d['hausdorff_angle'] = float(ha)
                # mesh_d['mean_angle'] = float(ma)
                mesh_d['test_mse'] = float(test_mse)
                mesh_d['mesh_mse'] = float(mesh_mse)

                new_d.append({'mesh': mesh_d})
            except:
                failed += 1
    
    print(f"Failed to compute metrics for {failed} meshes")
    return new_d

def unique_filename(save_directory, stub, ext, return_uuid=False):
    """
    Generate a unique file name using uuid from stub and append the file extension.

    Parameters
    ----------
    save_directory : str
        Path to directory where the file will be saved
    stub : str
        Non-unqiue identifier of the type stored in the file
    ext : str
        File extension
    return_uuid : bool
        Return the uuid for the generated file.

    Returns
    -------
    fn : str
        Path to unique file.
    uuid : float, optional
        The unique identifier.
    """
    uid = uuid.uuid4()
    fn = os.path.join(save_directory, 
                        f"{stub}_{uid}.{ext.split('.')[-1]}")
    if return_uuid:
        return fn, uid
    return fn

def evaluate_structure(test_d, test_shape, pp, td, psf_width, mpc, no):
    """
    Inner part of test_structure() loop, abstracted for multiprocessing

    Parameters
    ----------
    test_d : dict
        List of testing parameters
    test_shape : shape.Shape
        Structure for generating test shape
    pp : float
        Point cloud density
    td : float
        Threshold density for this point cloud density
    psf_width : tuple or list
        Width of psf for a single localization
    mpc : int
        Mean number of photons per localization
    no : float
        Fraction of localizations which are noise
    """
    # start_time = time.strftime('%Y%d%m_%HH%M')

    # generate and save the points
    points_fp = unique_filename(test_d['save_fp'], 'points', 'hdf')
    points_ds, points_md = generate_smlm_pointcloud_from_shape(test_shape, density=test_d['point_cloud']['density'], 
                                                    p=pp, 
                                                    psf_width=psf_width, 
                                                    mean_photon_count=mpc,
                                                    bg_photon_count=test_d['system']['bg_photon_count'], 
                                                    noise_fraction=no, save_fn=points_fp)

    # reload the generated points as a data source
    # points_ds = HDFSource(points_fp, 'Data')
    
    sw_md = []
    iso_md = []
    for spn in test_d['shrinkwrapping']['samplespernode']:
        # Generate an isosurface, where we set the initial density based on the ground truth density
        iso_save_fp = unique_filename(test_d['save_fp'], 'isosurface', 'stl')
        initial_mesh, i_md = generate_coarse_isosurface(points_ds,
                                                        samples_per_node=spn, 
                                                        threshold_density=td,  # test_d['shrinkwrapping']['density'][i], #test_d['point_cloud']['density']*test_d['point_cloud']['p']/(10*spn),  # choose a density less than the point cloud density 
                                                        smooth_curvature=True, 
                                                        repair=False, 
                                                        remesh=True, 
                                                        keep_largest=True, 
                                                        save_fn=iso_save_fp)
        
        # Compute shrinkwrapping isosurfaces
        s_md = test_shrinkwrap(initial_mesh, points_ds, test_d['shrinkwrapping']['max_iters'], test_d['shrinkwrapping']['step_size'], 
                            test_d['shrinkwrapping']['search_rad'], test_d['shrinkwrapping']['remesh_every'], 
                            test_d['shrinkwrapping']['search_k'], save_folder=test_d['save_fp'])
        for s in s_md:
            s['mesh']['samplespernode'] = spn
        iso_md.append({'isosurface': i_md})
        sw_md.extend(s_md)
    
    # Compute screened poisson reconstruction isosurfaces
    spr_md = test_spr(points_ds, test_d['screened_poisson']['max_iters'], test_d['screened_poisson']['search_k'],
                    test_d['screened_poisson']['depth'], test_d['screened_poisson']['samplespernode'], 
                    test_d['screened_poisson']['pointweight'], save_folder=test_d['save_fp'])
    
    # Save the results
    yaml_out, uid = unique_filename(test_d['save_fp'], 'run', 'yaml', return_uuid=True)
    with open(yaml_out, 'w') as f:
        yaml.safe_dump([{'points': points_md}, *iso_md, *sw_md, *spr_md], f)
    
    # Load the results and compute metrics
    res = compute_mesh_metrics(yaml_out, test_shape, psf_width=psf_width,
                               mean_photon_count=mpc,
                               bg_photon_count=test_d['system']['bg_photon_count'])
    
    # Save the results
    yaml_out = os.path.join(test_d['save_fp'], f"run_{uid}_metrics.yaml")
    with open(yaml_out, 'w') as f:
        yaml.safe_dump([{'points': points_md}, *iso_md, *res], f)

    return yaml_out

def test_structure(yaml_file, multiprocessing=False, force=False):
    
    with open(yaml_file) as f:
        test_d = yaml.safe_load(f)
    
    if not os.path.exists(test_d['save_fp']):
        os.mkdir(test_d['save_fp'])
    else:
        # let's check if any analyzed files exist in this folder
        import glob
        prev_runs = glob.glob(os.path.join(test_d['save_fp'], '*_metrics.yaml'))
        if len(prev_runs) > 0:  # There are files in here from a previous run
            if force:
                # Delete all of the files from the previous run
                prev_runs = glob.glob(os.path.join(test_d['save_fp'],'*'))
                for run in prev_runs:
                    os.remove(run)
            else:
                # we should attempt a graceful restart
                for run in prev_runs:
                    psf_widths = []
                    noise_fractions = []
                    mean_photon_counts = []
                    threshold_densities = []
                    point_densities = []
                    with open(run, 'r') as fp:
                        run_dict = yaml.safe_load(fp)
                        psf_widths.append(tuple(run_dict[0]['points']['psf_width']))
                        noise_fractions.append(run_dict[0]['points']['noise_fraction'])
                        mean_photon_counts.append(run_dict[0]['points']['mean_photon_count'])
                        threshold_densities.append(run_dict[1]['isosurface']['threshold_density'])
                        point_densities.append(run_dict[0]['points']['p'])

                    run_params = []
                    # loop over psf combinations, if present
                    for psf_width in psf_widths:
                        # allow for testing at multiple noise levels
                        for no in noise_fractions:
                            # test at multiple mean photon counts (multiple localization precisions)
                            for mpc in mean_photon_counts:
                                # test at multiple densities via adjustment in Monte-Carlo sampling
                                # enumerate to allow changing shrinkwrapping density with generated density
                                for td, pp in zip(threshold_densities, point_densities):
                                    run_params.append((pp, td, psf_width, mpc, no))
                    

    # Generate the theoretical shape
    test_shape = getattr(shape, test_d['shape']['type'])(**test_d['shape']['parameters'])

    # create the loop values
    psf_widths = itertools.product(test_d['system']['psf_width_x'], 
                                        test_d['system']['psf_width_y'], 
                                        test_d['system']['psf_width_z'])
    noise_fractions = test_d['point_cloud']['noise_fraction']
    mean_photon_counts = test_d['system']['mean_photon_count']
    threshold_densities = test_d['shrinkwrapping']['density']
    point_densities = test_d['point_cloud']['p']

    if np.isscalar(noise_fractions):
        noise_fractions = [noise_fractions]
    if np.isscalar(mean_photon_counts):
        mean_photon_counts = [mean_photon_counts]
    if np.isscalar(threshold_densities):
        threshold_densities = [threshold_densities]
    if np.isscalar(point_densities):
        point_densities = [point_densities]

    params = []
    # loop over psf combinations, if present
    for psf_width in psf_widths:
        # allow for testing at multiple noise levels
        for no in noise_fractions:
            # test at multiple mean photon counts (multiple localization precisions)
            for mpc in mean_photon_counts:
                # test at multiple densities via adjustment in Monte-Carlo sampling
                # enumerate to allow changing shrinkwrapping density with generated density
                for td, pp in zip(threshold_densities, point_densities):
                    params.append((pp, td, psf_width, mpc, no))

    if 'run_params' in locals():
        print("Run params: ", run_params)
        print("Total params:", params)
        params = list(set(params)-set(run_params))
    
    print("Diffed params: ", params)

    if multiprocessing:
        import multiprocessing as mp

        with mp.Pool() as pool:
            yaml_out = pool.starmap(partial(evaluate_structure, test_d, test_shape), params)

    else:
        for p in params:
            yaml_out = partial(evaluate_structure, test_d, test_shape)(*p)

    return yaml_out

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Load a YAML.')
    parser.add_argument('filename', help="YAML file containing paramater space for evaluation.", 
                        default=None, nargs='?')
    parser.add_argument('-mp', '--multiprocessing', help="Parallelize evaluation process.", 
                        dest='mp', default=False, action='store_true')
    parser.add_argument('-f', '--force', help="Restart evaluation from the beginning.",
                        dest='force', default=False, action='store_true')

    args = parser.parse_args()

    test_structure(args.filename, multiprocessing=args.mp, force=args.force)

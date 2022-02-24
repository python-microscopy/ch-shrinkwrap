"""
Tools for comparing mesh data sets to theoertical object they are
approximating. Motivation for ordered pairs, Hausdorff, mean distance 
and smoothness functions comes from 

Berger, Matt, Josh Levine, Luis Gustavo Nonato, Gabriel Taubin, and 
Claudio T. Silva. "An End-to-End Framework for Evaluating Surface 
Reconstruction." Technical. SCI Institute, University of Utah, 2011. 
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.190.6244.

"""

from . import _membrane_mesh as membrane_mesh
from . import util

from PYME.simulation.locify import points_from_sdf
from PYME.experimental.isosurface import distance_to_mesh

import os
import time
import math
import scipy.spatial
import numpy as np
import pymeshlab as ml

def points_from_mesh(mesh, dx_min=1, p=0.1, return_normals=False):
    """
    Generate random uniform sampling of points on a mesh.

    mesh : ch_shrinkwrap._membrane_mesh.MembraneMesh
        Mesh representation of an object.
    dx_min : float
        The target side length of a voxel. Sets maximum sampling rate.
    p : float
        Monte-Carlo acceptance probability.
    """

    def mesh_sdf(pts):
        return distance_to_mesh(pts.T, mesh)
    
    xl, yl, zl, xu, yu, zu = mesh.bbox
    diag = math.sqrt((xu-xl)*(xu-xl)+(yu-yl)*(yu-yl)+(zu-zl)*(zu-zl))
    centre = ((xl+xu)/2, (yl+yu)/2, (zl+zu)/2)

    d = points_from_sdf(mesh_sdf, diag/2, centre, dx_min=dx_min, p=p)

    if return_normals:
        # TODO: This repeats a fair bit of distance_to_mesh

        # Create a list of face centroids for search
        face_centers = mesh._vertices['position'][mesh.faces].mean(1)

        # Construct a kdtree over the face centers
        tree = scipy.spatial.cKDTree(face_centers)

        _, _faces = tree.query(d.T, k=1)

        normals = mesh._faces['normal'][mesh._faces['halfedge'] != -1][_faces]

        return d.T, normals

    return d.T

def construct_ordered_pairs(o, m, no, nm, dx_max=1, k=10, special_case=True):
    """
    Find pairs between point sets omega (o) and m s.t.

    For (ox, oa), if oa \in o then ox = \Phi(oa) \in m 
    where \Phi(oa) = oa + d*N(oa) where d is the signed
    distance along the normal N(oa) of the point oa.

    For (ma, mx), if mx \in m then ma = \Psi(mx) \in o
    where \Psi(mx) = mx + g*N(mx) where g is the signed
    distance along the normal N(mx) of the point mx.

    See section 6 of Berger et al.

    Parameters
    ----------
    o : np.array
        N x 3 point data set
    m : np.array
        M x 3 point data set
    no : np.array
        N x 3 array of normals for o
    nm : np.array
        M x 3 array of normals for m
    dx_max : float
        Maximum distance \Phi or \Psi can be from an 
        actual data point (sampling rate of uniformly
        sampled point set)
    k : int
        Up to how many nearest neighbors should we consider?
        
    Returns
    -------
        ox : np.array
            Indices of x in the omega set
        oa : np.array
            alpha in the omega set
        mx : np.array
            x in the M set
        ma : np.array
            alpha in the M set
    """

    # construct a kdtree 
    otree = scipy.spatial.cKDTree(o)
    mtree = scipy.spatial.cKDTree(m)

    # Get the closest point for each m
    # and vice versa
    om, oi = otree.query(m, 1)  # (M,)  # for x in M, find alpha in o
    mo, mi = mtree.query(o, 1)  # (N,)  # for alpha in o, find x in M

    # Find the dot product of normal and the vector from
    # each set to the other set
    mdot = (nm*(o[oi]-m)).sum(1) # (M,)  # candidate mapping psi(x)
    odot = (no*(m[mi]-o)).sum(1) # (N,)  # candidate mapping phi(alpha)

    # If the dot product is larger than if the vectors o[oi]-m and m+g*N(m)
    # were displaced tip to tip by dx_max, keep them (o[oi]-m is within
    # dx_max of m+g*N(m))
    mop = om - dx_max*dx_max/(2*om)
    omp = mo - dx_max*dx_max/(2*mo)
    mdot_bool = np.abs(mdot) > mop
    odot_bool = np.abs(odot) > omp
    mdot_idxs = np.flatnonzero(mdot_bool) 
    odot_idxs = np.flatnonzero(odot_bool)

    # Clean up duplicates
    ox, ox_inds = np.unique(mi[odot_idxs], return_index=True)
    oa = odot_idxs[ox_inds]
    ma, ma_inds = np.unique(oi[mdot_idxs], return_index=True)
    mx = mdot_idxs[ma_inds]

    if special_case:
        #########
        # For any points that don't pass the check, expand the 
        # search. Find the closest point that does pass the check.
        m2, o2 = m[~mdot_bool], o[~odot_bool]
        om2, oi2 = otree.query(m2, k)
        mo2, mi2 = mtree.query(o2, k)

        mdot2 = ((nm[~mdot_bool])[:,None,:]*(o[oi2]-m2[:,None,:])).sum(2) # (M,k)  # candidate mapping psi(x)
        odot2 = ((no[~odot_bool])[:,None,:]*(m[mi2]-o2[:,None,:])).sum(2) # (N,k)  # candidate mapping phi(alpha)
        mop2 = om2 - dx_max*dx_max/(2*om2+1e6)
        omp2 = mo2 - dx_max*dx_max/(2*mo2+1e6)
        mdot_bool2 = np.abs(mdot2) > mop2
        odot_bool2 = np.abs(odot2) > omp2

        mdot_idxs2 = mi2[np.arange(len(mi2)),np.argmax(odot_bool2, axis=1)].squeeze()
        odot_idxs2 = oi2[np.arange(len(oi2)),np.argmax(mdot_bool2, axis=1)].squeeze()

        # Throw way indices that did not meet the condition anywhere
        mdot_idxs2 = mdot_idxs2[np.sum(odot_bool2, axis=1)>0]
        odot_idxs2 = odot_idxs2[np.sum(mdot_bool2, axis=1)>0]

        # Then, add that point and its closest point in
        # this set to the correspondence set. See Figure 10 of 
        # Berger et al.
        _, oi3 = otree.query(m[mdot_idxs2], 1)
        _, mi3 = mtree.query(o[odot_idxs2], 1)

        # Clean up duplicates
        ox2, ox_inds2 = np.unique(mi3, return_index=True)
        oa2 = odot_idxs2[ox_inds2]
        ma2, ma_inds2 = np.unique(oi3, return_index=True)
        mx2 = mdot_idxs2[ma_inds2]

        #####

        # Intersection

        # One more unique call, as the special case may create a second mapping on an 
        # already existing mapping.
        oa2_inds = np.isin(oa2,oa)  # defined by mapping from a to x, throw away accordingly
        mx2_inds = np.isin(mx2,mx)

        # Stack them
        ox = np.hstack([ox, ox2[~oa2_inds]])
        oa = np.hstack([oa, oa2[~oa2_inds]])
        mx = np.hstack([mx, mx2[~mx2_inds]])
        ma = np.hstack([ma, ma2[~mx2_inds]])

    return ox, oa, mx, ma

def mean_and_hausdorff_distance_from_ordered_pairs(o, m, ox, oa, mx, ma):
    dist_o = np.linalg.norm(o[oa] - m[ox], axis=1)
    dist_m = np.linalg.norm(o[ma] - m[mx], axis=1)

    hausdorff = max(np.max(dist_o), np.max(dist_m))
    mean = 0.5*(np.mean(dist_o) + np.mean(dist_m))

    return hausdorff, mean

def mean_and_hausdorff_smoothness_from_ordered_pairs(no, nm, ox, oa, mx, ma):
    angle_o = np.arccos((no[oa]*nm[ox]).sum(1))
    angle_m = np.arccos((no[ma]*nm[mx]).sum(1))

    hausdorff = max(np.max(angle_o), np.max(angle_m))
    mean = 0.5*(np.mean(angle_o) + np.mean(angle_m))

    return hausdorff, mean

def test_points_mesh_stats(points, normals, mesh, dx_min=5, p=1.0):
    # Generate a set of test points from this mesh
    mesh_points, mesh_normals = points_from_mesh(mesh, 
                                                 dx_min=dx_min, 
                                                 p=p, return_normals=True)
    
    # Compute ordered points between this mesh and test_points
    ox, oa, mx, ma = construct_ordered_pairs(points, mesh_points, 
                                             normals, mesh_normals, 
                                             dx_max=dx_min)
    
    # print(test_points.shape[0], mesh_points.shape[0], ox.shape[0], mx.shape[0])

    # Compute hausdorff and mean distance (nm) and smoothness (rad)
    hd, md = mean_and_hausdorff_distance_from_ordered_pairs(points, mesh_points, ox, oa, mx, ma)
    ha, aa = mean_and_hausdorff_smoothness_from_ordered_pairs(normals, mesh_normals, ox, oa, mx, ma)
    
    return hd, md, ha, aa


def generate_smlm_pointcloud_from_shape(test_shape, density=1, p=0.0001, psf_width=250.0, 
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
    cap_points = test_shape.points(density=density, p=p, psf_width=psf_width, 
                            mean_photon_count=mean_photon_count, 
                            resample=True)
    # find the precision of each simulated point
    cap_sigma = test_shape._sigma
    
    # simualte clusters at each of the points
    cap_points, cap_sigma = smlmify_points(cap_points, cap_sigma, psf_width=psf_width, 
                                           mean_photon_count=mean_photon_count, 
                                           **kw)

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
        noise_sigma = util.loc_error(noise_points.shape, model='poisson', 
                                    psf_width=psf_width, 
                                    mean_photon_count=mean_photon_count)
        
        # simulate clusters at each of the random noise points
        noise_points, noise_sigma = smlmify_points(noise_points, noise_sigma, psf_width=psf_width, 
                                                mean_photon_count=mean_photon_count)
        
        # stack the regular and noise points
        points = np.vstack([cap_points,noise_points])
        sigma = np.vstack([cap_sigma,noise_sigma])
        s = np.sqrt((sigma*sigma).sum(1))
    else:
        points = cap_points
        sigma = cap_sigma
        s = np.sqrt((sigma*sigma).sum(1))

    # pass metadata associated with this simulation
    md = {'shape': test_shape.__str__(), 'density': density, 'p': p, 'psf_width': psf_width, 
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
        
    return np.vstack([points.T,s]).T, md

def smlmify_points(points, sigma, psf_width=250.0, mean_photon_count=300.0, max_points_per_cluster=10, max_points=None):
    # simulate clusters of points around each noise point
    noise_points = np.vstack([np.random.normal(points, sigma) for i in range(max_points_per_cluster)])
    
    sz = points.shape[0] if max_points is None else max_points
    
    # extract only sz points, some of the originals may disappear
    noise_points = noise_points[np.random.choice(np.arange(noise_points.shape[0]), size=sz, replace=False)]
    
    # Generate new sigma for each of these points
    noise_sigma = util.loc_error(noise_points.shape, model='poisson', 
                                 psf_width=psf_width, 
                                 mean_photon_count=mean_photon_count)
    
    return noise_points, noise_sigma

def generate_coarse_isosurface(ds, samples_per_node=1,
                               threshold_density=2e-5, smooth_curvature=True, 
                               repair=False, remesh=True, cull_inner_surfaces=True, 
                               keep_largest=True, save_fn=None):
    from PYME.experimental.octree import gen_octree_from_points
    from PYME.experimental import dual_marching_cubes
    from PYME.experimental import _triangle_mesh as triangle_mesh
    
    ot = gen_octree_from_points(ds)
    
    dmc = dual_marching_cubes.PiecewiseDualMarchingCubes(threshold_density)
    dmc.set_octree(ot.truncate_at_n_points(samples_per_node))
    tris = dmc.march(dual_march=False)
    
    surf = triangle_mesh.TriangleMesh.from_np_stl(tris, 
                                                  smooth_curvature=smooth_curvature)
    
    if repair:
        surf.repair()

    if remesh:
        surf.remesh()

    if keep_largest:
        surf.keep_largest_connected_component()
    elif cull_inner_surfaces:
        surf.remove_inner_surfaces()
        
    md = {'samples_per_node': samples_per_node, 'threshold_density': threshold_density,
          'smooth_curvature': smooth_curvature, 'repair': repair, 'remesh': remesh, 
          'cull_inner_surfaces': cull_inner_surfaces}
    
    if save_fn is not None:
        surf.to_stl(save_fn)
        md['filename'] = save_fn
        
    return surf, md

def screened_poisson(points, k=10, smoothiter=0,
                     flipflag=False, viewpos=[0,0,0], depth=8, fulldepth=5,
                     cgdepth=0, scale=1.1, samplespernode=1.5, pointweight=4, 
                     iters=8, confidence=False, preclean=False, save_fn=None):
    """
    Run screened poisson reconstruction on a set of points, using meshlab.
    
    For more information on these parameters, see meshlab.
    
    Parameters
    ----------
    points : np.array
        (M,3) array 
    see meshlab
    
    Returns
    -------
    str
        File path to STL of reconstruction
        
    """

    mesh = ml.Mesh(points)

    ms = ml.MeshSet()  # create a mesh
    ms.add_mesh(mesh)

    start = time.time()
    # compute normals
    ms.compute_normals_for_point_sets(k=k,  # number of neighbors
                                      smoothiter=smoothiter,
                                      flipflag=flipflag,
                                      viewpos=viewpos)
    # run SPR
    ms.surface_reconstruction_screened_poisson(visiblelayer=False,
                                               depth=depth,
                                               fulldepth=fulldepth,
                                               cgdepth=cgdepth,
                                               scale=scale,
                                               samplespernode=samplespernode,
                                               pointweight=pointweight,
                                               iters=iters,
                                               confidence=confidence,
                                               preclean=preclean)
    stop = time.time()
    duration = stop-start

    md = {'type': 'spr', 'k': int(k), 'smoothiter': bool(smoothiter), 'flipflag': bool(flipflag), 'viewpos': list(viewpos),
          'depth': int(depth), 'fulldepth': int(fulldepth), 'cgdepth': int(cgdepth), 'scale': float(scale),
          'samplespernode': float(samplespernode), 'pointweight': float(pointweight), 'iters': int(iters),
          'confidence': bool(confidence), 'preclean': bool(preclean), 'duration': float(duration)}

    if save_fn is not None:
        ms.save_current_mesh(file_name=save_fn, unify_vertices=True)
        md['filename'] = save_fn
    
    return (ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()), md

def test_shrinkwrap(mesh, ds, max_iters, step_size, search_rad, remesh_every, search_k, save_folder=None):
    points = np.vstack([ds['x'], ds['y'], ds['z']]).T
    sigma = ds['sigma']
    
    failed_count = 0
    md = []
    for it in max_iters:
        for lam in step_size:
            for sr in search_rad:
                for re in remesh_every:
                    for k in search_k:
                        # Copy the mesh over
                        mesh = membrane_mesh.MembraneMesh(mesh=mesh)

                        # set params
                        mesh.max_iter = it
                        mesh.step_size = lam
                        mesh.search_k = k
                        mesh.search_rad = sr
                        mesh.remesh_frequency = re
                        mesh.delaunay_remesh_frequency = 0

                        try:
                            start = time.time()
                            mesh.shrink_wrap(points, sigma, method='conjugate_gradient')
                            stop = time.time()
                            duration = stop-start
                            mmd = ({'type': 'shrinkwrap', 'iterations': int(it), 'remesh_every': int(re), 'lambda': float(lam), 
                            'search_k': int(k), 'search_rad': float(sr), 'ntriangles': int(mesh.faces.shape[0]), 'duration': float(duration)})
                            if save_folder is not None:
                                wrap_fp = os.path.join(save_folder,'_'.join(f"sw_mesh_{time.time():.1f}".split('.'))+".stl")
                                mesh.to_stl(wrap_fp)
                                mmd['filename'] = wrap_fp
                            md.append({'mesh': mmd})
                        except:
                            failed_count += 1
    print(f'{failed_count} shrinkwrapped meshes failed.')
    return md

def test_spr(ds, max_iters, search_k, depth, samplespernode, pointweight, save_folder=None):
    points = np.vstack([ds['x'], ds['y'], ds['z']]).T
    md = []
    failed_count = 0
    for it in max_iters:
        for k in search_k:
            for d in depth:
                for spn in samplespernode:
                    for wt in pointweight:
                        try:
                            wrap_fp = os.path.join(save_folder,'_'.join(f"spr_mesh_{time.time():.1f}".split('.'))+".stl")
                            _, mmd = screened_poisson(points, k=k, depth=d, samplespernode=spn, pointweight=wt,
                                                        iters=it, save_fn=wrap_fp)
                            md.append({'mesh': mmd})
                        except:
                            failed_count += 1
    print(f'{failed_count} SPR meshes failed.')
    return md

def compute_mesh_metrics(yaml_file, test_shape, dx_min=5, p=1.0, psf_width=250.0, mean_photon_count=300.0):
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
    import yaml
    import ch_shrinkwrap._membrane_mesh as membrane_mesh

    d, new_d = [], []
    with open(yaml_file) as f:
        d = yaml.safe_load(f)
    
    test_points, test_normals = test_shape.points(density=1.0/(dx_min**3), p=1.0, 
                                           psf_width=psf_width, 
                                           mean_photon_count=mean_photon_count, 
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
                vecs = mesh._vertices[mesh.faces]['position']
                errors = test_shape.sdf(vecs.mean(1).T)
                mse = np.nansum(errors**2)/len(errors)

                # Calculate distance and angle stats
                hd, md, ha, ma = test_points_mesh_stats(test_points, 
                                                        test_normals, 
                                                        mesh)
                
                mesh_d['mse'] = float(mse)
                mesh_d['hausdorff_distance'] = float(hd)
                mesh_d['mean_distance'] = float(md)
                mesh_d['hausdorff_angle'] = float(ha)
                mesh_d['mean_angle'] = float(ma)

                new_d.append({'mesh': mesh_d})
            except:
                failed += 1
                
    print(f"Failed to compute metrics for {failed} meshes")
    return new_d

def test_structure(yaml_file):
    from . import shape
    from PYME.IO.tabular import HDFSource
    
    import itertools
    import yaml
    import time
    
    with open('test.yaml') as f:
        test_d = yaml.safe_load(f)
    
    if not os.path.exists(test_d['save_fp']):
        os.mkdir(test_d['save_fp'])

    # Generate the theoretical shape
    test_shape = getattr(shape, test_d['shape']['type'])(**test_d['shape']['parameters'])
    
    # loop over psf combinations, if present
    for psf_width in itertools.product(test_d['system']['psf_width_x'], 
                                       test_d['system']['psf_width_y'], 
                                       test_d['system']['psf_width_z']):
        # allow for testing at multiple noise levels
        for no in test_d['point_cloud']['noise_fraction']:
            start_time = time.strftime('%Y%d%m_%HH%M')

            # generate and save the points
            points_fp = os.path.join(test_d['save_fp'], '_'.join(f"points_{time.time():.1f}".split('.'))+".hdf")
            _, points_md = generate_smlm_pointcloud_from_shape(test_shape, density=test_d['point_cloud']['density'], 
                                                               p=test_d['point_cloud']['p'], 
                                                               psf_width=psf_width, 
                                                               mean_photon_count=test_d['system']['mean_photon_count'], 
                                                               noise_fraction=no, save_fn=points_fp)

            # reload the generated points as a data source
            points_ds = HDFSource(points_fp, 'Data')
            
            sw_md = []
            iso_md = []
            for spn in test_d['shrinkwrapping']['samplespernode']:
                # Generate an isosurface, where we set the initial density based on the ground truth density
                initial_mesh, i_md = generate_coarse_isosurface(points_ds,
                                                                samples_per_node=spn, 
                                                                threshold_density=test_d['point_cloud']['density']*test_d['point_cloud']['p']/(10*spn), 
                                                                smooth_curvature=True, 
                                                                repair=False, 
                                                                remesh=True, 
                                                                keep_largest=True, 
                                                                save_fn=os.path.join(test_d['save_fp'], '_'.join(f"isosurface_{time.time():.1f}".split('.'))+".stl"))
                
                # Compute shrinkwrapping isosurfaces
                s_md = test_shrinkwrap(initial_mesh, points_ds, test_d['shrinkwrapping']['max_iters'], test_d['shrinkwrapping']['step_size'], 
                                       test_d['shrinkwrapping']['search_rad'], test_d['shrinkwrapping']['remesh_every'], 
                                       test_d['shrinkwrapping']['search_k'], save_folder=test_d['save_fp'])
                s_md['samplespernode'] = spn
                i_md = {'isosurface': i_md}
                iso_md.extend(i_md)
                sw_md.extend(s_md)
            
            # Compute screened poisson reconstruction isosurfaces
            spr_md = test_spr(points_ds, test_d['screened_poisson']['max_iters'], test_d['screened_poisson']['search_k'],
                              test_d['screened_poisson']['depth'], test_d['screened_poisson']['samplespernode'], 
                              test_d['screened_poisson']['pointweight'], save_folder=test_d['save_fp'])
            
            # Save the results
            yaml_out = os.path.join(test_d['save_fp'], f'run_{start_time}.yaml')
            with open(yaml_out, 'w') as f:
                yaml.safe_dump([{'points': points_md}, *iso_md, *sw_md, *spr_md], f)
            
            # Load the results and compute metrics
            res = compute_mesh_metrics(yaml_out, test_shape, psf_width=psf_width, mean_photon_count=test_d['system']['mean_photon_count'])
            
            # Save the results
            yaml_out = os.path.join(test_d['save_fp'], f'run_{start_time}_metrics.yaml')
            with open(yaml_out, 'w') as f:
                yaml.safe_dump([{'points': points_md}, {'isosurface': iso_md}, *res], f)

    return yaml_out

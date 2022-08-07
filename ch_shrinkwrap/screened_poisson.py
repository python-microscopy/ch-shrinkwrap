# pymeshlab must be compiled against a different version of OpenMP
# (libomp) than the one used in Conda (llvm-openmp)
import sys
if sys.platform == 'darwin':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from typing import Optional, Tuple

import pymeshlab as ml
import numpy.typing as npt

def screened_poisson(points : npt.ArrayLike, 
                     normals : Optional[npt.ArrayLike] = None,
                     k : int = 10,
                     smoothiter : int = 0,
                     flipflag : bool = False,
                     viewpos : npt.ArrayLike = [0,0,0],
                     visiblelayer : bool = False,
                     depth : int = 12,
                     fulldepth : int = 5,
                     cgdepth : int = 0,
                     scale : float = 1.1,
                     samplespernode : float = 1.5,
                     pointweight : float = 4,
                     iters : int = 8,
                     confidence : bool = False,
                     preclean : bool = False,
                     threads : int = 8) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    This surface reconstruction algorithm creates watertight surfaces from oriented point sets.

    See: 
    - https://pymeshlab.readthedocs.io/en/latest/filter_list.html#compute_normal_for_point_clouds
    - https://pymeshlab.readthedocs.io/en/latest/filter_list.html#generate_surface_reconstruction_screened_poisson
    
    Parameters
    ----------
    points : npt.ArrayLike
        (M, (x,y,z)) array of points
    normals : npt.ArrayLike
        (M, (x,y,z)) array of normals for each point
    k : int
        Neighbour num: The number of neighbors used to estimate normals.
    smoothiter : int
        Smooth Iteration: The number of smoothing iteration done on the p used to estimate and propagate normals.
    flipflag : bool
        Flip normals w.r.t. viewpoint: If the 'viewpoint' (i.e. scanner position) is known, it can be used to 
        disambiguate normals orientation, so that all the normals will be oriented in the same direction.
    viewpos : np.ndarray[np.float64[3]]
        Viewpoint Pos.: The viewpoint position can be set by hand (i.e. getting the current viewpoint) or it 
        can be retrieved from mesh camera, if the viewpoint position is stored there.
    visiblelayer : bool
        Merge all visible layers: Enabling this flag means that all the visible layers will be used for providing 
        the points.
    depth : int
        Reconstruction Depth: This integer is the maximum depth of the tree that will be used for surface 
        reconstruction. Running at depth d corresponds to solving on a voxel grid whose resolution is no larger 
        than 2^d x 2^d x 2^d. Note that since the reconstructor adapts the octree to the sampling density, the 
        specified reconstruction depth is only an upper bound. The default value for this parameter is 8.
    fulldepth : int
        Adaptive Octree Depth: This integer specifies the depth beyond depth the octree will be adapted. At 
        coarser depths, the octree will be complete, containing all 2^d x 2^d x 2^d nodes. The default value 
        for this parameter is 5.
    cgdepth : int
        Conjugate Gradients Depth: This integer is the depth up to which a conjugate-gradients solver will be 
        used to solve the linear system. Beyond this depth Gauss-Seidel relaxation will be used. The default 
        value for this parameter is 0.
    scale : float
        Scale Factor: This floating point value specifies the ratio between the diameter of the cube used for 
        reconstruction and the diameter of the samples' bounding cube. The default value is 1.1.
    samplespernode : float
        Minimum Number of Samples: This floating point value specifies the minimum number of sample points that 
        should fall within an octree node as the octree construction is adapted to sampling density. For 
        noise-free samples, small values in the range [1.0 - 5.0] can be used. For more noisy samples, larger 
        values in the range [15.0 - 20.0] may be needed to provide a smoother, noise-reduced, reconstruction. 
        The default value is 1.5.
    pointweight : float
        Interpolation Weight: This floating point value specifies the importants that interpolation of the point 
        samples is given in the formulation of the screened Poisson equation. The results of the original 
        (unscreened) Poisson Reconstruction can be obtained by setting this value to 0. The default value for 
        this parameter is 4.
    iters : int
        Gauss-Seidel Relaxations: This integer value specifies the number of Gauss-Seidel relaxations to be 
        performed at each level of the hierarchy. The default value for this parameter is 8.
    confidence : bool
        Confidence Flag: Enabling this flag tells the reconstructor to use the quality as confidence 
        information; this is done by scaling the unit normals with the quality values. When the flag is not 
        enabled, all normals are normalized to have unit-length prior to reconstruction.
    preclean : bool
        Pre-Clean: Enabling this flag force a cleaning pre-pass on the data removing all unreferenced vertices 
        or vertices with null normals.
    threads : int
        Number of threads to use for mesh calculation.

    Returns
    -------
    npt.ArrayLike
        Mesh vertices
    npt.ArrayLike
        Mesh faces
    """

    if normals is not None:
        mesh = ml.Mesh(vertex_matrix=points,
                       v_normals_matrix=normals)

        ms = ml.MeshSet()  # create a mesh
        ms.add_mesh(mesh)
    else:
        mesh = ml.Mesh(vertex_matrix=points)

        ms = ml.MeshSet()
        ms.add_mesh(mesh)

        # compute normals
        ms.compute_normal_for_point_clouds(k=k,
                                           smoothiter=smoothiter,
                                           flipflag=flipflag,
                                           viewpos=viewpos)

    # run SPR
    ms.generate_surface_reconstruction_screened_poisson(visiblelayer=visiblelayer,
                                                        depth=depth,
                                                        fulldepth=fulldepth,
                                                        cgdepth=cgdepth,
                                                        scale=scale,
                                                        samplespernode=samplespernode,
                                                        pointweight=pointweight,
                                                        iters=iters,
                                                        confidence=confidence,
                                                        preclean=preclean,
                                                        threads=threads)

    return (ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix())
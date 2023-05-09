import numpy as np

def masked_distance_to_mesh(points, surf, smooth=False, smooth_k=0.1, tree=None, face_mask=None):
    """
    Calculate the distance to a mesh from points in a tabular dataset 

    Parameters
    ----------
        points : np.array
            3D point cloud to fit (nm).
        surf : PYME.experimental._triangle_mesh
            Isosurface
        smooth : bool
            Smooth distance to mesh?
        smooth_k : float
            Smoothing constant, by default 0.1 = smoothed by 1/10 nm
        tree : scipy.spatial.cKDTree
            Optionally pass a tree of face centers to use if calling
            this function multiple times.
    """
    from PYME.experimental.isosurface import sdf_min, triangle_sdf

    if face_mask is None:
        face_mask = np.ones(surf.faces.shape, dtype='bool')
   
    if tree is None:
        import scipy.spatial

        # Create a list of face centroids for search
        face_centers = surf._vertices['position'][surf.faces[face_mask]].mean(1)

        # Construct a kdtree over the face centers
        tree = scipy.spatial.cKDTree(face_centers)

    # Get M closet face centroids for each point
    M = 5
    _, _faces = tree.query(points, k=M, workers=-1)
    
    # Get position representation
    _v = surf.faces[face_mask][_faces]
    v = surf._vertices['position'][_v]  # (n_points, M, (v0,v1,v2), (x,y,z))

    # the negative reverses the norpa sign flip
    # trick 2 from https://iquilezles.org/www/articles/interiordistance/interiordistance.htm
    return -sdf_min(triangle_sdf(points, v), smooth=smooth, k=smooth_k)


def _masked_intersection_sdf(points, surf, face_mask, offset=10.0, tree=None, tree_c=None):
    from PYME.experimental.isosurface import distance_to_mesh

    d1 = distance_to_mesh(points, surf, tree=tree)
    d2 = masked_distance_to_mesh(points, surf, face_mask=face_mask, tree=tree_c)

    return np.maximum(d1 - d2 - offset, d1)


def punch_holes(m, offset=10.0, pi_threshold=0):
    from PYME.experimental import func_octree
    from PYME.experimental import dual_marching_cubes
    from PYME.experimental import _triangle_mesh as triangle_mesh
    from scipy.spatial import cKDTree

    face_mask = m.point_influence[m.faces].max(1) > pi_threshold

    face_centers = m._vertices['position'][m.faces].mean(1)
    # Construct a kdtree over the face centers
    tree = cKDTree(face_centers)

    face_centers = m._vertices['position'][m.faces[face_mask]].mean(1)
    # Construct a kdtree over the face centers
    tree_c = cKDTree(face_centers)

    def f(x, y, z):
        return _masked_intersection_sdf(np.vstack([x, y, z]).T, m, face_mask, offset, tree=tree, tree_c=tree_c)

    x0, y0, z0, x1, y1, z1 = m.bbox

    o = func_octree.FOctree([x0, x1, y0, y1, z0, z1], f, maxdepth=6)
    mc = dual_marching_cubes.PiecewiseDualMarchingCubes()
    mc.set_octree(o)
    tris = mc.march()
    surf = triangle_mesh.TriangleMesh.from_np_stl(tris, smooth_curvature=True)
    #surf.repair()
    surf.remesh()
    return surf


def wrap_start(points, offset=10., neighbourhood=50):
    from PYME.experimental import func_octree
    from PYME.experimental import dual_marching_cubes
    from PYME.experimental import _triangle_mesh as triangle_mesh
    from scipy.spatial import cKDTree

    tree = cKDTree(points)

    def f(x, y, z):
        pts = np.vstack([x, y, z]).T
        dd, ii = tree.query(pts, k=neighbourhood, workers=-1)
        return dd.max(1) - offset

    x0, y0, z0 = points.min(0)
    x1, y1, z1 = points.max(0)

    o = func_octree.FOctree([x0, x1, y0, y1, z0, z1], f, maxdepth=6)
    mc = dual_marching_cubes.PiecewiseDualMarchingCubes()
    mc.set_octree(o)
    tris = mc.march()
    surf = triangle_mesh.TriangleMesh.from_np_stl(tris, smooth_curvature=True)
    
    #surf.repair()
    surf.remesh()
    return surf


    


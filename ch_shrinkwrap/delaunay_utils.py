import numpy as np
import scipy.spatial

from PYME.experimental import isosurface

from ch_shrinkwrap import sdf

def orient_simps(d, v):
    """
    Ensure all simplex vertices are wound such that 
    tris_from_delaunay(d, oriented=True) returns 
    triangles with normals pointing away from 
    simplex centroids.
    
    Parameters
    ----------
        d : scipy.spatial.qhull.Delaunay or np.array
            Delaunay triangulation or an (N,4) array 
            of simplices
        v : np.array
            (M,3) array of x,y,z coordinates of vertices
            in Delaunay triangulation d
            
    Returns
    -------
        tri : np.array
            (N, 4) array of oriented simplices
    """
    if isinstance(d, scipy.spatial.qhull.Delaunay):
        d = d.simplices
    # For each simplex, check if one ordered triangle's normal is pointing toward 
    #   or away from the centroid
    v_tri = v[d]
    centroid = v_tri.mean(1)
    v21 = v_tri[:,1,:]-v_tri[:,2,:]
    v23 = v_tri[:,3,:]-v_tri[:,2,:]
    n123 = np.cross(v23,v21,axis=1)
    orientation = np.sign((n123*(v_tri[:,1,:]-centroid)).sum(1))
    
    # If it's pointing away, flip a single edge. The triangle should now be oriented.
    tri = d
    orientation_mask = orientation == -1
    tmp_3 = np.copy(tri[orientation_mask,3])
    tri[orientation_mask,3] = tri[orientation_mask,2]
    tri[orientation_mask,2] = tmp_3
    
    return tri

def tris_from_delaunay(d, return_index=False, oriented=False):
    """
    Get a list of triangles from a Delaunay triangulation.
    
    Parameters
    ----------
        d : scipy.spatial.qhull.Delaunay or np.array
            Delaunay triangulation or an (N,4) array 
            of simplices
        return_index : bool
            Return the index of the simplex associated with 
            each triangle
        oriented : bool
            Return triangles in winding such that normals 
            point out of tetrahedron
            
    Returns
    -------
        tris : np.array
            (M, 3) array of triangles with indices corresponding 
            to the same vertices as the Delaunay triangulation
    """
    if isinstance(d, scipy.spatial.qhull.Delaunay):
        d = d.simplices
    if oriented:
        # Return triangles so normals point out of tetrahedron
        tris = np.vstack([d[:,[2,1,0]], d[:,1:], d[:,[3,2,0]], d[:,[0,1,3]]])
    else:
        # Return order such that it is easy to find unique triangles
        tris = np.vstack([d[:,:3], d[:,1:], d[:,[0,2,3]], d[:,[0,1,3]]])
    if return_index:
        inds = np.hstack(4*[np.arange(d.shape[0])])
        return tris, inds
    return tris

def surf_from_delaunay(d):
    """
    Find the surface of the Delaunay triangulation.
    
    Parameters
    ----------
        d : scipy.spatial.qhull.Delaunay or np.array
            Delaunay triangulation or an (N,4) array 
            of simplices
            
    Returns
    -------
        np.array
            (M, 3) array of valence one triangles with indices 
            corresponding to the same vertices as the Delaunay 
            triangulation
    """
    tris = tris_from_delaunay(d, oriented=True)
    _, inds, counts = np.unique(np.sort(tris, axis=1),axis=0,return_index=True,return_counts=True)
    # Extract the valence one faces, ordered
    return tris[inds[counts==1]]

def del_simps(d, inds):
    """
    Delete simplices from a Delaunay triangulation d.
    
    Parameters
    ----------
        d : scipy.spatial.qhull.Delaunay or np.array
                Delaunay triangulation or an (N,4) array 
                of simplices
        inds : np.array or list
            Indices of simplices to delete
    
    Returns
    -------
        np.array
            (M, 4) array of simplices with inds simplices
            removed from d.
    """
    if isinstance(d, scipy.spatial.qhull.Delaunay):
        d = d.simplices
    d_mask = np.ones(d.shape[0], dtype=bool)
    d_mask[inds] = False
    return d[d_mask,:]
    
def ext_simps(d, mesh):
    """
    Find which simplices in d are outside a mesh. Assumes
    d triangulates mesh vertices.
    
    Parameters
    ----------
        d : scipy.spatial.qhull.Delaunay or np.array
            Delaunay triangulation or an (N,4) array 
            of simplices
        mesh : PYME.experimental._triangle_mesh.TriangleMesh
            
    Returns
    -------
        tri : np.array
            (N, 4) array of oriented simplices
    """

    if isinstance(d, scipy.spatial.qhull.Delaunay):
        d = d.simplices
    
    # Remove all simplices with centers outside the original mesh
    v = mesh._vertices['position'][mesh._vertices['halfedge']!=-1]
    simp_centers = np.mean(v[d],axis=1)
    simp_dist = isosurface.distance_to_mesh(simp_centers, mesh)
    return np.where(simp_dist>0)[0]

def empty_simps(d, v, pts, eps=0.0):
    """
    Find which simplices in d do not contain any
    3D coordinates in pts.
    
    Parameters
    ----------
        d : scipy.spatial.qhull.Delaunay or np.array
            Delaunay triangulation or an (N,4) array 
            of simplices
        v : np.array
            (M,3) array of x,y,z coordinates of vertices
            in Delaunay triangulation d
        pts : np.array
            (L,3) array of x,y,z coordinates of points
            to test 
            
    Returns
    -------
        np.array
            Indices of empty simplices in d.
    """
    if isinstance(d, scipy.spatial.qhull.Delaunay):
        d = d.simplices
    d_mask = np.zeros(d.shape[0], dtype=bool)

    # Loop over the simplices
    # Loop should be optimized with a KDTree:
    #     # Spatially separate all points in the dataset
    #     tree = scipy.spatial.cKDTree(pts)
    # 
    #     centroid = vs.mean(0)
    #     diameter = (np.max(((vs[:,None]-vs[None,:])**2).sum(2)))**0.5
    #     neighbors = tree.query_ball_point(centroid, diameter)
    #     np.all(tetrahedron(pts[neighbors], *vs)>eps)
    # Keeping the KDTree stuff out for now to make it easier to debug
    for _i in range(d.shape[0]):
        _vs = d[_i,:]
        vs = v[_vs]
        # If all the original points lie outside this tetrahedron,
        # remove this tetrahedron
        if np.all(sdf.tetrahedron(pts, *vs)>eps):
            d_mask[_i] = True
            
    return np.where(d_mask)[0]

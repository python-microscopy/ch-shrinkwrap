from math import sqrt
import numpy as np
import scipy.spatial

from PYME.experimental import isosurface

from . import sdf, util

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
    orientation_mask = (orientation == -1)
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
        # tris = np.vstack([d[:,[2,1,0]], d[:,1:], d[:,[3,2,0]], d[:,[0,1,3]]])
        tris = np.vstack([d[:,[0,1,2]], d[:,[1,3,2]], d[:,[3,0,2]], d[:,[0,3,1]]])
    else:
        # Return order such that it is easy to find unique triangles
        tris = np.vstack([d[:,:3], d[:,1:], d[:,[0,2,3]], d[:,[0,1,3]]])
    if return_index:
        inds = np.hstack(4*[np.arange(d.shape[0])])
        return tris, inds
    return tris

def surf_from_delaunay(d, oriented=True):
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
    tris = tris_from_delaunay(d, oriented=oriented)
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
    from PYME.experimental.isosurface import distance_to_mesh

    if isinstance(d, scipy.spatial.qhull.Delaunay):
        d = d.simplices
    
    # Remove all simplices with centers outside the original mesh
    v = mesh._vertices['position'][mesh._vertices['halfedge']!=-1]
    simp_centers = np.mean(v[d],axis=1)
    simp_dist = distance_to_mesh(simp_centers, mesh)
    return np.flatnonzero(simp_dist > 0)

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
        eps : float
            Multiplier of circumradius 
            
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

        # v30 = vs[0,:] - vs[3,:]  # DA
        # v31 = vs[1,:] - vs[3,:]  # DB
        # v32 = vs[2,:] - vs[3,:]  # DC
        # a2 = util.fast_3x3_cross(v31,v32)

        # V = (1.0/6.0)*abs((v30*a2).sum())
        # a = sqrt((v30*v30).sum())
        # b = sqrt((v31*v31).sum())
        # c = sqrt((v32*v32).sum())
        # v12 = vs[2,:] - vs[1,:]
        # v20 = vs[0,:] - vs[2,:]
        # v01 = vs[1,:] - vs[0,:]
        # A = sqrt((v12*v12).sum()) # BC
        # B = sqrt((v20*v20).sum()) # CA
        # C = sqrt((v01*v01).sum()) # AB
        # aA = a*A
        # bB = b*B
        # cC = c*C
        # t0 = aA+bB+cC
        # t1 = aA+bB-cC
        # t2 = aA-bB+cC
        # t3 = bB+cC-aA
        # R = sqrt(t0*t1*t2*t3)/(24*V)  # circumradius

        # a1 = util.fast_3x3_cross(v30,v31)
        # a3 = util.fast_3x3_cross(v32,v30)
        # a4 = util.fast_3x3_cross(-v01,v12)
        # A1 = 0.5*sqrt((a1*a1).sum())
        # A2 = 0.5*sqrt((a2*a2).sum())
        # A3 = 0.5*sqrt((a3*a3).sum())
        # A4 = 0.5*sqrt((a4*a4).sum())
        # r = 3*V/(A1+A2+A3+A4)  # inradius

        # print('R: {} r: {} R-r: {}'.format(R,r,(R-r)))

        # If all the original points lie outside this tetrahedron,
        # remove this tetrahedron
        n_inside = np.sum(sdf.tetrahedron(pts, *vs)<=eps)
        # print('n_inside: {}'.format(n_inside))
        if n_inside == 0:
            d_mask[_i] = True
            
    return np.where(d_mask)[0]

def voronoi_poles(vor, point_normals):
    """
    Compute the positive and negative Voronoi poles in vor. The poles are the
    furthest points (on either side of ) on the Voronoi diagram from each of the original points.
    
    -1 index indicates a point at infinity.
    
    Amenta, N, and M Bern. "Surface Reconstruction by Voronoi Filtering." 
    Discrete & Computational Geometry 22 (1999): 481-504.
    
    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Voronoi diagram 
    point_normals: np.array
        Normals for vor.points, used for Voronoi poles on the convex hull.
        NOTE: this is a bit of a shortcut, and should be the average normal
        of the convex hull faces. Can lead to wonkiness.
        
    Returns
    -------
    p_pos : np.array
        Index of positive Vonoroni pole for corresponding point in vor.vertices
    p_neg : np.array
        Index of negative Voronoi pole for corresponding point in vor.vertices
    """
    # pregame
    sz = len(vor.point_region)
    p_pos, p_neg = np.zeros(sz,dtype=int), np.zeros(sz,dtype=int)
    
    # loop over the points we constructed a Voronoi diagram of
    for i, reg in enumerate(vor.point_region):
        cell_points = vor.regions[reg]
        dn = vor.vertices[cell_points] - vor.points[i][None,:]
        d = np.linalg.norm(dn,axis=1)
        
        if cell_points[0] == -1:
            # we are on the convex hull
            cell_points, dn, d = cell_points[1:], dn[1:,:], d[1:]
            p_pos[i] = -1
            pn = point_normals[i]
        else:
            # pick the Voronoi vertex furthest from this point
            di = np.argmax(d)
            p_pos[i] = cell_points[di]
            pn = dn[di,:]

        # negative pole is furthest vertex with negative dot product
        # between vector to vertex and vector to positive pole
        s = ((pn*dn).sum(1) < 1)
        p_neg[i] = cell_points[np.argmax(s*d)]
    
    return p_pos, p_neg

def clean_neg_voronoi_poles(mesh, np):
    # Make sure we have no negative voronoi poles outside of the mesh
    # TODO: This shouldn't be necessary, all negative voronoi poles
    # should lie on the medial axis of the mesh.

    from PYME.experimental.isosurface import distance_to_mesh
    d = distance_to_mesh(np, mesh)
    return np[d < 0.0,:]

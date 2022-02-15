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
    simp_dist = distance_to_mesh(simp_centers, mesh, smooth=False)
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

CORNER_ANGLE = 3*np.pi/2

def remove_singular_faces(faces, v):
    # precompute normals
    v1 = v[faces[:,1]]
    a = (v[faces[:,0]]-v1)
    b = (v[faces[:,2]]-v1)
    norms = np.cross(a,b,axis=1)
    nn = np.linalg.norm(norms,axis=1)
    norms /= nn[:,None]
    norms[nn==0] = 0  # this should not happen

    # Find unique edge occurances
    edges = np.zeros((3*faces.shape[0],2),dtype=faces.dtype)
    edges[::3] = faces[:,[0,1]]
    edges[1::3] = faces[:,[1,2]]
    edges[2::3] = faces[:,[2,0]]
    edges = np.sort(edges,axis=1)
    _, idxs, counts = np.unique(edges,axis=0,return_counts=True,return_inverse=True)

    kept_faces = np.ones(faces.shape[0], dtype=bool)
    
    # Remove slivers
    a_norm = np.linalg.norm(a,axis=1)
    b_norm = np.linalg.norm(b,axis=1)
    ab_norm = np.linalg.norm(a-b,axis=1)
    circumradius = a_norm*b_norm*ab_norm/(2*nn)
    area = np.pi*circumradius*circumradius
    adiff = 0.5*nn/area
    # remove triangles that have 10% of the area of their circumcircle
    kept_faces[adiff < 0.1] = False
    
    # Delete any unshared faces
    for i in np.flatnonzero(counts==1):
        candidate_faces = np.flatnonzero(idxs==i)//3
        kept_faces[candidate_faces[0]] = False
    
    # Delete any faces with sharp angles between them
    for i in np.flatnonzero(counts==2):
        candidate_faces = np.flatnonzero(idxs==i)//3
        dot = np.abs((norms[candidate_faces[0],:]*norms[candidate_faces[1],:]).sum())
        if np.arccos(dot) > CORNER_ANGLE:
            kept_faces[candidate_faces[0]] = False
            kept_faces[candidate_faces[1]] = False

    # Loop over edges with more than 2 incident faces
    for i in np.flatnonzero(counts>2):
        # grab their faces
        candidate_faces = np.flatnonzero(idxs==i)//3
        
        # compute the average normal of each face and its neighbors
#         candidate_face_norms = np.zeros((candidate_faces.shape[0],3),dtype=float)
#         for j, f in enumerate(candidate_faces):
#             edge0, edge1, edge2 = np.sort(np.vstack([faces[f,[0,1]],
#                                                      faces[f,[1,2]],
#                                                      faces[f,[2,0]]]),
#                                           axis=1)

#             # Check which faces contain these edges
#             norm_faces = np.unique(np.flatnonzero((edges==edge0).prod(1) | (edges==edge1).prod(1) | (edges==edge2).prod(1))//3)
#             candidate_face_norms[j,:] = np.mean(norms[norm_faces,:],axis=0)
#             print(norms[f,:], candidate_face_norms[j,:])

        # compare them and keep the two that have the least
        # sharp angles
        max_dot = -2
        _kept_face0 = 0
        _kept_face1 = 0
        for jj, j in enumerate(candidate_faces):
            for kk, k in enumerate(candidate_faces):
                if j==k:
                    continue
                dot = np.abs((norms[j,:]*norms[k,:]).sum())
                # dot = np.abs((candidate_face_norms[jj,:]*candidate_face_norms[kk,:]).sum())
                if dot > max_dot:
                    max_dot = dot
                    _kept_face0 = j
                    _kept_face1 = k
                    
        if max_dot == -2:
            print("max_dot is still -2??")
                    
        if np.arccos(max_dot) > CORNER_ANGLE:
            # Delete everything
            for j in candidate_faces:
                kept_faces[j] = False
        else:
            for j in candidate_faces:
                if (j == _kept_face0) or (j == _kept_face1):
                    continue
                kept_faces[j] = False
            
    return faces[kept_faces,:]

def construct_outer_surface(faces, v, starting_face=0):
    """
    Compute the outer surface from candidate faces.
    """
    # precompute normals
    v1 = v[faces[:,1]]
    a = (v[faces[:,0]]-v1)
    b = (v[faces[:,2]]-v1)
    norms = np.cross(a,b,axis=1)
    nn = np.linalg.norm(norms,axis=1)
    norms /= nn[:,None]
    norms[nn==0] = 0  # this should not happen

    # Find unique edge occurances
    edges = np.zeros((3*faces.shape[0],2),dtype=faces.dtype)
    edges[::3] = faces[:,[0,1]]
    edges[1::3] = faces[:,[1,2]]
    edges[2::3] = faces[:,[2,0]]
    edges = np.sort(edges,axis=1)
    unique_edges, edge_idxs, edge_counts = np.unique(edges,axis=0,return_counts=True,return_inverse=True)
    edge_inds = dict()
    for i in range(unique_edges.shape[0]):
        edge_inds[tuple(unique_edges[i,:])] = i

    visited_faces = np.zeros(faces.shape[0], dtype=bool)
    kept_edges = np.zeros(edge_idxs.shape[0], dtype=int)
    kept_faces = np.zeros(faces.shape[0], dtype=bool)
    
    faces_to_visit = [starting_face]
    
    while faces_to_visit:
        curr_face = faces_to_visit.pop()
        #print(curr_face)
        
        if visited_faces[curr_face]:
            # we've already been here
            continue
        
        # Label that we've been here
        visited_faces[curr_face] = True
        
        # Look at the current edges
        edge0, edge1, edge2 = np.sort(np.vstack([faces[curr_face,[0,1]],
                                                  faces[curr_face,[1,2]],
                                                  faces[curr_face,[2,0]]]),
                                            axis=1)
        edge0_idx, edge1_idx, edge2_idx = edge_inds.get(tuple(edge0)), edge_inds.get(tuple(edge1)), edge_inds.get(tuple(edge2))
        
        if (edge_counts[edge0_idx] == 1) or (edge_counts[edge1_idx] == 1) or (edge_counts[edge2_idx] == 1):
            # Don't add a face that doesn't connect to any other faces
            continue
            
        # Test if this face is going to create a singularity
        if (kept_edges[edge0_idx] == 2) or (kept_edges[edge1_idx] == 2) or (kept_edges[edge2_idx] == 2):
            continue
            
        # No? Add it
        kept_faces[curr_face] = True
        kept_edges[edge0_idx] += 1
        kept_edges[edge1_idx] += 1
        kept_edges[edge2_idx] += 1
        
        # Add adjacent faces
        add_faces(curr_face, edge0_idx, edge_idxs, edge_counts, faces_to_visit, norms)
        add_faces(curr_face, edge1_idx, edge_idxs, edge_counts, faces_to_visit, norms)
        add_faces(curr_face, edge2_idx, edge_idxs, edge_counts, faces_to_visit, norms)
        
    return faces[kept_faces]

def add_faces(curr_face, edge_idx, edge_idxs, edge_counts, faces_to_visit, norms):
    if (edge_counts[edge_idx] == 1):
        #print("one edge?")
        return
    elif (edge_counts[edge_idx] == 2):
        candidate_faces = np.flatnonzero(edge_idxs==edge_idx)//3
        #print(candidate_faces)
        
        # Don't add corners
        dot = np.abs((norms[candidate_faces[0],:]*norms[candidate_faces[1],:]).sum())
        if np.arccos(dot) > CORNER_ANGLE:
            #print("No corner")
            return
        
        # Add the adjacent face
        if candidate_faces[0] == curr_face:
            faces_to_visit.append(candidate_faces[1])
        else:
            faces_to_visit.append(candidate_faces[0])
    else:
        # edge counts > 2
        candidate_faces = np.flatnonzero(edge_idxs==edge_idx)//3
        
        # Keep the smoothest transition between this face and the candidate faces
        max_dot = -2
        _kept_face = 0
        for i in range(len(candidate_faces)):
            if candidate_faces[i] == curr_face:
                continue
            dot = np.abs((norms[curr_face,:]*norms[candidate_faces[i],:]).sum())
            if dot > max_dot:
                max_dot = dot
                _kept_face = candidate_faces[i]
                    
        # Don't add corners
        if np.arccos(max_dot) > CORNER_ANGLE:
            #print("No corner")
            return
        
        faces_to_visit.append(_kept_face)

def sliver_simps(d, v, sigma0=0.1, rho0=0.1):
    """
    Find the simplicies making up slivers.
    
    Li, “SLIVER-FREE THREE DIMENSIONAL DELAUNAY MESH GENERATION.”
    
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
    
    v_tri = v[d]

    v21 = v_tri[:,1,:]-v_tri[:,2,:]
    nv21 = np.linalg.norm(v21,axis=1)
    v23 = v_tri[:,3,:]-v_tri[:,2,:]
    nv23 = np.linalg.norm(v23,axis=1)
    v20 = v_tri[:,0,:]-v_tri[:,2,:]
    nv20 = np.linalg.norm(v20,axis=1)
    v30 = v_tri[:,0,:]-v_tri[:,3,:]
    nv30 = np.linalg.norm(v30,axis=1)
    v10 = v_tri[:,0,:]-v_tri[:,1,:]
    nv10 = np.linalg.norm(v10,axis=1)
    v13 = v_tri[:,1,:]-v_tri[:,3,:]
    nv13 = np.linalg.norm(v13,axis=1)
    
    aA = nv21*nv30
    bB = nv23*nv10
    cC = nv20*nv13
    
    V = (1.0/6.0)*np.abs((v21*np.cross(v23,v20),axis=1).sum(1))  # volume
    l = np.min([nv21,nv23,nv20,nv30,nv10,nv13])  # minimum edge length
    
    R = np.sqrt((aA+bB+cC)*(aA+bB-cC)*(aA-bB+cC)*(-aA+bB+cC))/(24*V) # circumradius
    
    sigma = V/(l*l*l)  # shape quality
    
    rho = R/l  # radius-edge ratio
    
    return d[(sigma<sigma0)&(rho<rho0)]
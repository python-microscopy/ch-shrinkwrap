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

from PYME.simulation.locify import points_from_sdf
from PYME.experimental.isosurface import distance_to_mesh

import math
import scipy.spatial
import numpy as np

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

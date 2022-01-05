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

def points_from_mesh(mesh, dx_min=1, p=0.1, normals=False):
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

    if normals:
        # TODO: This repeats a fair bit of distance_to_mesh

        # Create a list of face centroids for search
        face_centers = mesh._vertices['position'][mesh.faces].mean(1)

        # Construct a kdtree over the face centers
        tree = scipy.spatial.cKDTree(face_centers)

        _, _faces = tree.query(d.T, k=1)

        normals = mesh._faces['normal'][mesh._faces['halfedge'] != -1][_faces]

        return d.T, normals

    return d.T

def constructed_ordered_pairs(o, m, no, nm, dx_max=1, K=5):
    """
    Find pairs between point sets omega (o) and m s.t.

    For (ox, oa), if oa \in o then ox = \Phi(oa) \in m 
    where \Phi(oa) = oa + d*N(oa) where d is the signed
    distance along the normal N(oa) of the point oa.

    For (ma, mx), if mx \in m then ma = \Psi(mx) \in o
    where \Psi(mx) = mx + g*N(mx) where g is the signed
    distance along the normal N(mx) of the point mx.

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
    K : int
        How many closest points should we consider when
        looking for pair correlations?
    """

    # construct a kdtree 
    otree = scipy.spatial.cKDTree(o)
    mtree = scipy.spatial.cKDTree(m)

    # Get the K closest o points for each m
    # and vice versa
    _, oi = otree.query(m, k=K)  # (M, K)
    _, mi = mtree.query(o, k=K)  # (N, K)

    # Find the dot product of the unit normal 
    # with the vector between point sets
    om = o[oi] - m[:,None,:]  # (M, K, 3)
    mo = m[mi] - o[:,None,:]  # (N, K, 3)
    mdot = (nm[:,None,:]*om).sum(2) # (M, K)
    odot = (no[:,None,:]*mo).sum(2) # (N, K)

    # Find the dot product larger than ||o[oi]-m|| - dx_max^2/(2*||o[oi]-m||)
    om_norm = (om*om).sum(2)
    mo_norm = (mo*mo).sum(2)
    mdot_idxs = np.argmax(mdot, 1) - (~np.any(mdot, axis=1))  # -1 if not found
    odot_idxs = np.argmax(odot, 1) - (~np.any(odot, axis=1))

    
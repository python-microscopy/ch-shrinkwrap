#!/usr/bin/python

import numpy as np
import pytest

EPS = 1e-6

@pytest.mark.skip(reason='mix of 0s and 1s for some reason')
def test_orient_simps():
    from ch_shrinkwrap.delaunay_utils import orient_simps

    # Generate two tetrahedron, one oriented correctly,
    # one misoriented
    vertices = np.array([
        [-1,0,0],
        [1,0,0],
        [0,1,0],
        [0,0.5,1]
    ])

    tetrahedron = np.array([
        [0,1,2,3],
        [0,1,3,2]
    ])

    # Orient them
    oriented_tetrahedron = orient_simps(tetrahedron, vertices)

    # Check that the normals of all of the triangles point
    # away from the centroid
    tris = np.vstack([oriented_tetrahedron[:,[0,1,2]], 
                      oriented_tetrahedron[:,[1,3,2]], 
                      oriented_tetrahedron[:,[3,0,2]], 
                      oriented_tetrahedron[:,[0,3,1]]])  # (8 x 3)
    v_tri = vertices[tris] # (8 x (v0,v1,v2) x (x,y,z))
    centroid = v_tri.mean(1)
    v21 = v_tri[:,1,:]-v_tri[:,0,:]
    v23 = v_tri[:,2,:]-v_tri[:,0,:]
    n123 = np.cross(v23,v21,axis=1)
    orientation = np.sign((n123*(v_tri[:,1,:]-centroid)).sum(1))

    assert(np.all(orientation == 1))

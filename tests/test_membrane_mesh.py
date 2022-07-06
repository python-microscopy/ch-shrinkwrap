#!/usr/bin/python

import numpy as np

EPS = 1e-6

def spherical_mesh(R=1, n_subdivision=3):
    # Generate a spherical mesh from icosahedral subdivision

    from PYME.Analysis.points.spherical_harmonics import icosahedron_mesh
    from PYME.Analysis.points.coordinate_tools import spherical_to_cartesian

    from ch_shrinkwrap import _membrane_mesh as membrane_mesh

    # Quasi-regular sample points on a unit sphere with icosahedron subdivision
    azimuth, zenith, f = icosahedron_mesh(n_subdivision)

    x, y, z = spherical_to_cartesian(azimuth, zenith, R)
    v = np.vstack([x,y,z]).T

    return membrane_mesh.MembraneMesh(v,f)

def planar_mesh(a=1, n_subdivision=1):
    # Generate a plane from subdividing a square
    from ch_shrinkwrap import _membrane_mesh as membrane_mesh
    
    step = 1/n_subdivision
    p = np.arange(0,1+step,step)
    pv = np.vstack([p]*(n_subdivision+1))*a
    x = pv.ravel('F')
    y = pv.ravel('C')
    z = np.zeros_like(x)
    v = np.vstack([x,y,z]).T

    lr = np.ravel(np.arange(len(p)-1)[None,:] + np.arange(len(p),len(x),len(p))[:,None],order='C')
    ll = np.ravel(np.arange(len(p)-1)[None,:] + np.arange(0,len(x)-len(p),len(p))[:,None],order='C')
    ul = np.ravel(np.arange(1,len(p))[None,:] + np.arange(0,len(x)-len(p),len(p))[:,None],order='C')
    ur = np.ravel(np.arange(1,len(p))[None,:] + np.arange(len(p),len(x),len(p))[:,None],order='C')
    f = np.vstack([np.vstack([ll,lr,ur]).T,np.vstack([ll,ur,ul]).T])

    return membrane_mesh.MembraneMesh(v,f)

def test_mean_curvature_plane():
    a = int(100*np.random.rand()+1)
    n_subdivision = int(5*np.random.rand()+1)

    mesh = planar_mesh(a, n_subdivision)
    # mesh.calculate_curvatures()

    assert np.abs(np.nanmean(mesh.curvature_mean)) < EPS


def test_mean_curvature_sphere():
    # confirm sphere of radius R has mean curvature of 1/R
    R = int(100*np.random.rand()+1)
    n_subdivision = int(4*np.random.rand()+2)

    mesh = spherical_mesh(R, n_subdivision)
    # mesh.calculate_curvatures()

    # TODO: there are many ways to get at this and decimal=1 is awfully permissive.
    #       this is a result of a few vertices getting up to 0.5 out of whack every
    #       few runs of this test.
    np.testing.assert_almost_equal(mesh.curvature_mean,1/R,decimal=2)

def test_gaussian_curvature_plane():
    a = int(100*np.random.rand()+1)
    n_subdivision = int(4*np.random.rand()+2)

    mesh = planar_mesh(a, n_subdivision)
    # mesh.calculate_curvatures()

    assert np.abs(np.nanmedian(mesh.curvature_gaussian)) < EPS


def test_gaussian_curvature_sphere():
    # confirm sphere of radius R has Gaussian curvature of 1/R**2
    R = int(100*np.random.rand()+1)
    n_subdivision = int(4*np.random.rand()+2)

    mesh = spherical_mesh(R, n_subdivision)
    # mesh.calculate_curvatures()

    # TODO: there are many ways to get at this and decimal=1 is awfully permissive.
    #       this is a result of a few vertices getting up to 0.05 out of whack every
    #       few runs of this test. this test still fails every once and a while on
    #       a radius of 1.
    np.testing.assert_almost_equal(mesh.curvature_gaussian,1/(R*R),decimal=4)

# def test_curvature_cyl():
#     # confirm cylinder of radius R has mean curvature 1/(2*R) and Gaussian curvature of 0
#     pass

# def test_curvature_grad_sphere():
#     # use known curvatures of sphere to 
#     pass
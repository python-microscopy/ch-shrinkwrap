#!/usr/bin/python

def test_curvature_sphere():
    # # confirm sphere of radius R has mean curvature of 1/R and Gaussian curvature of 1/R**2
    # from PYME.experimental import marching_cubes
    # from PYME.experimental import _triangle_mesh as triangle_mesh
    # from ch_shrinkwrap import membrane_mesh

    # R = 10

    # # Creates a sample sphere
    # S = marching_cubes.generate_sphere_image(R)
    # # Converts the sphere to vertices, indices (the equivalent for an image is position (x,y,z) and intensity)
    # v, i = marching_cubes.image_to_vertex_values(S, voxelsize=1.)

    # v -= 0.5*np.array(S.shape)[None, None, :]

    # threshold = 0.5
    # mc = marching_cubes.MarchingCubes(threshold)
    # mc.vertices = v
    # mc.values = i
    # tris = mc.march(dual_march=False)

    # mesh = triangle_mesh.TriangleMesh.from_np_stl(tris, smooth_curvature=True)
    # membrane = membrane_mesh.MembraneMesh(mesh=mesh)
    pass

# def test_curvature_cyl():
#     # confirm cylinder of radius R has mean curvature 1/(2*R) and Gaussian curvature of 0
#     pass

# def test_curvature_grad_sphere():
#     # use known curvatures of sphere to 
#     pass
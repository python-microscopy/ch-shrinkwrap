# Install the cgal package from conda-forge
# Tested with python 3.8.13/cgal 5.2.2

from CGAL import CGAL_Alpha_wrap_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Kernel import Point_3

import numpy as np
import numpy.typing as npt

def cgal_vertices_faces_triangle_mesh(Q: Polyhedron_3):
    vertices = np.zeros((Q.size_of_vertices(), 3), dtype=float)
    vertices_packed = {}
    faces = np.zeros((Q.size_of_facets(), 3), dtype=float)
    next_idx_v = 0
    for idx_f, facet in enumerate(Q.facets()):
        he = facet.halfedge()
        
        for j in range(3):
            p = he.vertex().point()
            v = tuple((p.x(), p.y(), p.z()))
            idx_v = vertices_packed.get(v, -1)
            if idx_v < 0:
                vertices[next_idx_v, :] = v
                vertices_packed[v] = next_idx_v
                idx_v = next_idx_v
                next_idx_v += 1
            faces[idx_f,j] = idx_v
            he = he.next()
            
    return vertices, faces

def alpha_wrap(points: npt.ArrayLike, alpha: float = 20.0, offset = 0.001):
    vertices_cgal = [Point_3(x, y, z) for x, y, z in points.astype(np.double)]
    Q = Polyhedron_3()
    CGAL_Alpha_wrap_3.alpha_wrap_3(vertices_cgal, alpha, offset, Q)

    nv, nf = cgal_vertices_faces_triangle_mesh(Q)

    return nv, nf

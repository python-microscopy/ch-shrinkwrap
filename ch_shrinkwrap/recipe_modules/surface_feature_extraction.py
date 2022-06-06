from PYME.IO import tabular, image, MetaDataHandler
from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, DictStrAny, CStr, Int, Bool, Float, Enum
import logging
from ch_shrinkwrap._membrane_mesh import DESCENT_METHODS

logger = logging.getLogger(__name__)

@register_module('SkeletonizeMembrane')
class SkeletonizeMembrane(ModuleBase):
    """
    Create a skeleton of a mesh using mean curvature flow.

    Tagliasacchi, Andrea, Ibraheem Alhashim, Matt Olson, and Hao Zhang. 
    "Mean Curvature Skeletons." Computer Graphics Forum 31, no. 5 
    (August 2012): 1735-44. https://doi.org/10.1111/j.1467-8659.2012.03178.x.
    """
    input = Input('surf')
    output = Output('skeleton')

    max_iters = Int(500)
    velocity_weight = Float(20.0)
    medial_axis_weight = Float(40.0)
    mesoskeleton = Bool(False)
    area_variation_factor = Float(0.0001)
    max_triangle_angle = Float(110.0)

    def execute(self, namespace):
        import numpy as np
        from ch_shrinkwrap import _membrane_mesh as membrane_mesh

        mesh = membrane_mesh.MembraneMesh(mesh=namespace[self.input], 
                                          max_iter=self.max_iters)

        # pts = np.ascontiguousarray(np.vstack([namespace[self.points]['x'], 
        #                                       namespace[self.points]['y'],
        #                                       namespace[self.points]['z']]).T)
        # try:
        #     sigma = namespace[self.points]['sigma']
        # except(KeyError):
        #     sigma = 10*np.ones_like(namespace[self.points]['x'])

        pts, sigma = None, None

        # Upsample to create better Voronoi poles
        l = 0.95*np.mean(mesh._halfedges['length'][mesh._halfedges['length'] != -1])
        mesh.remesh(target_edge_length=l)

        # Shrinkwrap membrane surface subject to curvature, velocity, and medial axis forces
        mesh.shrink_wrap(pts, sigma, method='skeleton', lam=[self.velocity_weight, 
                       self.medial_axis_weight], area_variation_factor=self.area_variation_factor,
                       max_triangle_angle=self.max_triangle_angle)

        if not self.mesoskeleton:
            collapse_count = 1
            while (collapse_count > 0):
                print(collapse_count)
                collapse_count = 0
                for i in range(mesh._vertices.shape[0]):
                    if mesh._vertices['halfedge'][i] == -1:
                        continue
                    # collapse the shortest edge on this vertex
                    n = mesh._vertices['neighbors'][i]
                    l = mesh._halfedges['length'][n[n!=-1]]
                    j = np.argmin(l)
                    collapse_ret = mesh.skeleton_edge_collapse(n[j])
                    collapse_count += collapse_ret

            # At this point we should be left with a set of edges defining the skeleton
            namespace[self.output] = mesh
        else:
            # return mesoskeleton mesh
            namespace[self.output] = mesh

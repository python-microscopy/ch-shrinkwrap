from PYME.IO import tabular, image, MetaDataHandler
from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, DictStrAny, CStr, Int, Bool, Float
import logging

logger = logging.getLogger(__name__)

# def estimate_density(ot):
#     """
#     Estimate the mean density of the octree.

#     Parameters
#     ----------
#         ot : PYME.experimental._octree.Octree
#             Octree
#     """
#     import numpy as np

#     max_depth = ot._nodes['depth'].max()
#     density_sc = 1.0/np.prod(ot.box_size(np.arange(max_depth + 1)), axis=0)
#     node_mask = (ot._nodes['nPoints'] != 0)
#     nodes_in_use = ot._nodes[node_mask]
#     density = nodes_in_use['nPoints']*density_sc[nodes_in_use['depth']]

#     return np.median(density)

@register_module('ShrinkwrapMembrane')
class ShrinkwrapMembrane(ModuleBase):
    input = Input('surf')
    ouput = Output('membrane')
    points = Input('filtered_localizations')

    max_iters = Int(100)
    step_size = Float(1)
    attraction_weight = Float(1)
    curvature_weight = Float(-1)
    largest_component_only = Bool(True)

    def execute(self, namespace):
        import copy
        import numpy as np
        from ch_shrinkwrap import membrane_mesh

        mesh = membrane_mesh.MembraneMesh(mesh=namespace[self.input])

        mesh.a = self.attraction_weight
        mesh.c = self.curvature_weight
        mesh.max_iter = self.max_iters
        mesh.step_size = self.step_size

        pts = np.vstack([namespace[self.points]['x'], 
                         namespace[self.points]['y'],
                         namespace[self.points]['z']]).T
        try:
            sigma = namespace[self.points]['sigma']
        except(KeyError):
            sigma = np.ones_like(namespace[self.points]['x'])

        if self.largest_component_only:
            mesh.keep_largest_connected_component()
        mesh.shrink_wrap(pts, sigma)

        namespace[self.output] = mesh
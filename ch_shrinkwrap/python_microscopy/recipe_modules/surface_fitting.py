from PYME.IO import tabular, image, MetaDataHandler
from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, DictStrAny, CStr, Int, Bool, Float, Enum
import logging
from ch_shrinkwrap.membrane_mesh import DESCENT_METHODS

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
    # attraction_weight = Float(1)
    # curvature_weight = Float(-1)
    largest_component_only = Bool(True)
    search_k = Int(200)
    temperature = Float(25.0)
    kc = Float(0.514)
    kg = Float(0.0)
    skip_prob = Float(0.0)
    method = Enum(DESCENT_METHODS)

    def execute(self, namespace):
        import numpy as np
        from ch_shrinkwrap import membrane_mesh

        mesh = membrane_mesh.MembraneMesh(mesh=namespace[self.input], 
                                          search_k=self.search_k,
                                          temp=self.temperature,
                                          kc=self.kc,
                                          kg=self.kg,
                                          max_iter=self.max_iters,
                                          step_size=self.step_size,
                                          skip_prob=self.skip_prob)

        pts = np.ascontiguousarray(np.vstack([namespace[self.points]['x'], 
                                              namespace[self.points]['y'],
                                              namespace[self.points]['z']]).T)
        try:
            sigma = namespace[self.points]['sigma']
        except(KeyError):
            sigma = np.ones_like(namespace[self.points]['x'])

        if self.largest_component_only:
            mesh.keep_largest_connected_component()
        # from PYME.util import mProfile
        # mProfile.profileOn(['membrane_mesh.py'])
        mesh.shrink_wrap(pts, sigma, method=self.method)
        # mProfile.profileOff()
        # mProfile.report()

        namespace[self.output] = mesh

from PYME.IO import tabular, image, MetaDataHandler
from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, DictStrAny, CStr, Int, Bool, Float, Enum
import logging
from ch_shrinkwrap._membrane_mesh import DESCENT_METHODS

logger = logging.getLogger(__name__)

@register_module('ShrinkwrapMembrane')
class ShrinkwrapMembrane(ModuleBase):
    input = Input('surf')
    output = Output('membrane')
    points = Input('filtered_localizations')

    max_iters = Int(100)
    step_size = Float(10.0)
    attraction_weight = Float(1)
    curvature_weight = Float(1)
    search_rad = Float(100.0)
    search_k = Int(20)
    kc = Float(0.514)
    kg = Float(-0.514)
    skip_prob = Float(0.0)
    remesh_frequency = Int(5)
    delaunay_remesh_frequency = Int(50)
    min_hole_radius = Int(100)
    sigma = CStr('sigma')
    method = Enum(DESCENT_METHODS)

    def execute(self, namespace):
        import numpy as np
        from ch_shrinkwrap import _membrane_mesh as membrane_mesh

        mesh = membrane_mesh.MembraneMesh(mesh=namespace[self.input], 
                                          search_k=self.search_k,
                                          kc=self.kc,
                                          kg=self.kg,
                                          max_iter=self.max_iters,
                                          step_size=self.step_size,
                                          skip_prob=self.skip_prob,
                                          remesh_frequency=self.remesh_frequency,
                                          delaunay_remesh_frequency=self.delaunay_remesh_frequency,
                                          delaunay_eps=self.min_hole_radius,
                                          a=self.attraction_weight,
                                          c=self.curvature_weight,
                                          search_rad=self.search_rad)

        namespace[self.output] = mesh

        pts = np.ascontiguousarray(np.vstack([namespace[self.points]['x'], 
                                              namespace[self.points]['y'],
                                              namespace[self.points]['z']]).T)
        try:
            sigma = namespace[self.points][self.sigma]
        except(KeyError):
            print(f"{self.sigma} not found in data source, defaulting to 10 nm precision.")
            sigma = 10*np.ones_like(namespace[self.points]['x'])

        # from PYME.util import mProfile
        # mProfile.profileOn(['membrane_mesh.py'])
        mesh.shrink_wrap(pts, sigma, method=self.method)
        # mProfile.profileOff()
        # mProfile.report()

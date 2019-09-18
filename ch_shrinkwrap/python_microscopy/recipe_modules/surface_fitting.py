from PYME.IO import tabular, image, MetaDataHandler
from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, DictStrAny, CStr, Int, Bool, Float
import logging

logger = logging.getLogger(__name__)

@register_module('MembraneDualMarchingCubes')
class MembraneDualMarchingCubes(ModuleBase):
    input = Input('octree')
    output = Output('mesh')
    
    threshold_density = Float(2e-5)
    n_points_min = Int(5) # lets us truncate on SNR
    
    repair = Bool(False)
    remesh = Bool(False)
    
    def execute(self, namespace):
        from PYME.experimental import dual_marching_cubes
        from ch_shrinkwrap import membrane_mesh
        
        dmc = dual_marching_cubes.PiecewiseDualMarchingCubes(self.threshold_density)
        dmc.set_octree(namespace[self.input].truncate_at_n_points(int(self.n_points_min)))
        tris = dmc.march(dual_march=False)

        surf = membrane_mesh.MembraneMesh.from_np_stl(tris)
        
        if self.repair:
            surf.repair()
            
        if self.remesh:
            surf.remesh(5, l=0.5, n_relax=10)
            
        namespace[self.output] = surf

@register_module('ShrinkwrapMembrane')
class ShrinkwrapMembrane(ModuleBase):
    membrane = Input('membrane')
    points = Input('input')

    max_iters = Int(100)
    step_size = Float(1)
    attraction_weight = Float(1)
    curvature_weight = Float(-1)

    def excecute(self, namespace):
        import numpy as np

        namespace[self.membrane].a = self.attraction_weight
        namespace[self.membrane].c = self.curvature_weight
        namespace[self.membrane].max_iter = self.max_iters
        namespace[self.membrane].step_size = self.step_size

        pts = np.vstack([namespace[self.points]['x'], 
                         namespace[self.points]['y'],
                         namespace[self.points]['z']]).T
        sigma = namespace[self.points]['sigma']

        namespace[self.membrane].shrink_wrap(pts, sigma)
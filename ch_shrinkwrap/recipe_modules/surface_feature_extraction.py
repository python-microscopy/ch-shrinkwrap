from PYME.IO import tabular, image, MetaDataHandler
from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, DictStrAny, CStr, Int, Bool, Float, Enum
import logging
from ch_shrinkwrap._membrane_mesh import DESCENT_METHODS

logger = logging.getLogger(__name__)

@register_module('SkeletonizeMembrane')
class SkeletonizeMembrane(ModuleBase):
    input = Input('surf')
    ouput = Output('skeleton')

    max_iters = Int(10)
    velocity_weight = Float(20.0)
    medial_axis_weight = Float(40.0)
    mesoskeleton = Bool(False)

    def execute(self, namespace):
        import numpy as np
        from ch_shrinkwrap import _membrane_mesh as membrane_mesh

        mesh = membrane_mesh.MembraneMesh(mesh=namespace[self.input], 
                                          max_iter=self.max_iters)

        pts = np.ascontiguousarray(np.vstack([namespace[self.points]['x'], 
                                              namespace[self.points]['y'],
                                              namespace[self.points]['z']]).T)
        try:
            sigma = namespace[self.points]['sigma']
        except(KeyError):
            sigma = 10*np.ones_like(namespace[self.points]['x'])

        # Shrinkwrap membrane surface subject to curvature, velocity, and medial axis forces
        mesh.shrink_wrap(pts, sigma, method='skeleton', lam=[self.velocity_weight, self.medial_axis_weight])

        if not self.mesoskeleton:
            # Once complete, collapse all edges that can be collapsed without making the mesh non-manifold
            xl = np.min(mesh._vertices['position'][mesh._vertices['halfedge']!=-1][:,0])
            xu = np.max(mesh._vertices['position'][mesh._vertices['halfedge']!=-1][:,0])
            yl = np.min(mesh._vertices['position'][mesh._vertices['halfedge']!=-1][:,1])
            yu = np.max(mesh._vertices['position'][mesh._vertices['halfedge']!=-1][:,1])
            zl = np.min(mesh._vertices['position'][mesh._vertices['halfedge']!=-1][:,2])
            zu = np.max(mesh._vertices['position'][mesh._vertices['halfedge']!=-1][:,2])
            bbox_diag_length = np.sqrt((xu-xl)*(xu-xl)+(yu-yl)*(yu-yl)+(zu-zl)*(zu-zl))
            collapse_threshold = 0.002*bbox_diag_length
            print(f"Diag length: {bbox_diag_length}  collapse_threshold: {collapse_threshold}")
            ct = mesh.collapse_edges(collapse_threshold)
            while (ct > 0):
                ct = mesh.collapse_edges(collapse_threshold)

            # At this point we should be left with a set of edges defining the skeleton

            namespace[self.output] = mesh
        else:
            # return mesoskeleton mesh
            namespace[self.output] = mesh

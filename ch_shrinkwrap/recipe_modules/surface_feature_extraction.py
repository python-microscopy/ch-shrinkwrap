from unicodedata import name
from PYME.IO import tabular, image, MetaDataHandler
from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, DictStrAny, CStr, Int, Bool, Float, Enum
import logging

from PYME.IO.MetaDataHandler import DictMDHandler

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
        from ch_shrinkwrap import _skeleton_mesh as membrane_mesh

        mesh = membrane_mesh.SkeletonMesh(mesh=namespace[self.input], 
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
        mesh.shrink_wrap(pts, sigma, method='skeleton', lam=[#self.velocity_weight, 
                       self.medial_axis_weight], area_variation_factor=self.area_variation_factor,
                       max_triangle_angle=self.max_triangle_angle)

        if not self.mesoskeleton:
            # collapse_count = 1
            # while (collapse_count > 0):
            #     collapse_count = 0
            #     for i in range(mesh._vertices.shape[0]):
            #         if mesh._vertices['halfedge'][i] == -1:
            #             continue
            #         # collapse the shortest edge on this vertex
            #         n = mesh._vertices['neighbors'][i]
            #         l = mesh._halfedges['length'][n[n!=-1]]
            #         j = np.argmin(l)
            #         collapse_ret = mesh.edge_collapse(n[j])
            #         collapse_count += collapse_ret

            # At this point we should be left with a set of edges defining the skeleton
            namespace[self.output] = mesh
        else:
            # return mesoskeleton mesh
            namespace[self.output] = mesh

@register_module('PointsFromMesh')
class PointsFromMesh(ModuleBase):
    input = Input('membrane0')
    output = Output('membrane0_localizations')

    dx_min = Float(5)
    p = Float(1.0)
    return_normals = Bool(True)

    def execute(self, namespace):
        from PYME.IO.tabular import DictSource
        from ch_shrinkwrap.evaluation_utils import points_from_mesh
        from PYME.IO import MetaDataHandler
        
        inp = namespace[self.input]
        md = MetaDataHandler.DictMDHandler(getattr(inp, 'mdh', None)) # get metadata from the input dataset if present
        points, normals = points_from_mesh(inp, dx_min=self.dx_min, p=self.p, 
                                           return_normals=self.return_normals)

        ds = DictSource({'x': points[:,0],
                         'y': points[:,1],
                         'z': points[:,2],
                         'xn': normals[:,0],
                         'yn': normals[:,1],
                         'zn': normals[:,2]})

        self._params_to_metadata(md)
        ds.mdh = md

        namespace[self.output] = ds

@register_module('AverageSquaredDistance')
class AverageSquaredDistance(ModuleBase):
    input = Input('filtered_localizations')
    input2 = Input('filtered')
    output = Output('average_squared_distance')

    def execute(self, namespace):
        import numpy as np
        from PYME.IO.tabular import DictSource
        from ch_shrinkwrap.evaluation_utils import average_squared_distance
        from PYME.IO import MetaDataHandler

        inp = namespace[self.input]
        inp2 = namespace[self.input2]
        md = MetaDataHandler.DictMDHandler(getattr(inp, 'mdh', None)) # get metadata from the input dataset if present
        md2 = MetaDataHandler.DictMDHandler(getattr(inp2, 'mdh', None)) # get metadata from the input dataset if present
        md.mergeEntriesFrom(md2)

        points0 = np.ascontiguousarray(np.vstack([inp['x'], 
                                                  inp['y'],
                                                  inp['z']]).T)

        points1 = np.ascontiguousarray(np.vstack([inp2['x'], 
                                                  inp2['y'],
                                                  inp2['z']]).T)

        mse0, mse1 = average_squared_distance(points0, points1)
        mse = np.sqrt((mse0+mse1)/2)

        ds = DictSource({'mse01': np.atleast_1d(mse0), 
                         'mse10': np.atleast_1d(mse1), 
                         'mse_rms': np.atleast_1d(mse)})
        self._params_to_metadata(md)
        ds.mdh = md

        namespace[self.output] = ds

@register_module('MeshProperties')
class MeshProperties(ModuleBase):
    inputMesh = Input('membrane')
    output = Output('mesh_props')

    def run(self, inputMesh):
        from PYME.IO.tabular import ColumnSource
        import numpy as np

        # calculate # of components
        # TODO - move this to mesh class
        comps = np.unique(inputMesh.component[inputMesh.vertex_mask])
        n_comps = len(comps)

        
        ds = ColumnSource(euler=np.atleast_1d(inputMesh.euler_characteristic),
                          genus=np.atleast_1d(inputMesh.genus),
                          manifold=np.atleast_1d(int(inputMesh.manifold)),
                          components=np.atleast_1d(n_comps),
                          #area=np.atleast_1d(np.sum([inputMesh.area(c) for c in comps])),
                          #volume=np.atleast_1d(np.sum([inputMesh.volume(c) for c in comps]))
                          )

        return ds

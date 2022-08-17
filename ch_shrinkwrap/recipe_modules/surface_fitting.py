from PYME.IO import tabular, image, MetaDataHandler
from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, DictStrAny, CStr, Int, Bool, Float, Enum
import logging
# from ch_shrinkwrap._membrane_mesh import DESCENT_METHODS

logger = logging.getLogger(__name__)

@register_module('ShrinkwrapMembrane')
class ShrinkwrapMembrane(ModuleBase):
    input = Input('surf')
    output = Output('membrane')
    points = Input('filtered_localizations')

    max_iters = Int(100)
    curvature_weight = Float(10.0)
    shrink_weight = Float(0)
    #attraction_weight = Float(1)
    #curvature_weight = Float(1)
    # search_rad = Float(100.0)
    # search_k = Int(20)
    kc = Float(1.0)  # Float(0.514)
    #kg = Float(-0.514)
    #skip_prob = Float(0.0)
    remesh_frequency = Int(5)
    cut_frequency = Int(0)
    min_hole_radius = Float(100.0)
    sigma_x = CStr('sigma_x')
    sigma_y = CStr('sigma_y')
    sigma_z = CStr('sigma_z')
    neck_theshold_low = Float(-1e-4)
    neck_threshold_high = Float(1e-2)
    neck_first_iter = Int(9)
    # method = Enum(DESCENT_METHODS)
    minimum_edge_length = Float(-1.0)

    def execute(self, namespace):
        import numpy as np
        from ch_shrinkwrap import _membrane_mesh as membrane_mesh

        mesh = membrane_mesh.MembraneMesh(mesh=namespace[self.input], 
                                        #   search_k=self.search_k,
                                          kc=self.kc,
                                          #kg=self.kg,
                                          max_iter=self.max_iters,
                                          step_size=self.curvature_weight,
                                          #skip_prob=self.skip_prob,
                                          remesh_frequency=self.remesh_frequency,
                                          delaunay_remesh_frequency=self.cut_frequency, # self.delaunay_remesh_frequency,
                                          delaunay_eps=self.min_hole_radius,
                                          neck_threshold_low = self.neck_theshold_low,
                                          neck_threshold_high = self.neck_threshold_high,
                                          neck_first_iter = self.neck_first_iter,
                                          shrink_weight = self.shrink_weight) # self.min_hole_radius)

                                          #a=self.attraction_weight,
                                          #c=self.curvature_weight,
                                        #   search_rad=self.search_rad)

        namespace[self.output] = mesh

        pts = np.ascontiguousarray(np.vstack([namespace[self.points]['x'], 
                                              namespace[self.points]['y'],
                                              namespace[self.points]['z']]).T)

        try:
            sigma = np.vstack([namespace[self.points][self.sigma_x],
                               namespace[self.points][self.sigma_y],
                               namespace[self.points][self.sigma_z]]).T
        except:
            try:
                sigma = namespace[self.points][self.sigma_x]
            except(KeyError):
                print(f"{self.sigma_x} not found in data source, defaulting to 10 nm precision.")
                sigma = 10*np.ones_like(namespace[self.points]['x'])

        # from PYME.util import mProfile
        # mProfile.profileOn(['membrane_mesh.py'])
        mesh.shrink_wrap(pts, sigma, method='conjugate_gradient', minimum_edge_length=self.minimum_edge_length)
        # mProfile.profileOff()
        # mProfile.report()


@register_module('ImageShrinkwrapMembrane')
class ImageShrinkwrapMembrane(ModuleBase):
    input = Input('surf')
    output = Output('membrane')
    input_image = Input('input')

    max_iters = Int(100)
    curvature_weight = Float(10.0)
    shrink_weight = Float(1.0)
    #attraction_weight = Float(1)
    #curvature_weight = Float(1)
    # search_rad = Float(100.0)
    # search_k = Int(20)
    kc = Float(1.0)  # Float(0.514)
    #kg = Float(-0.514)
    #skip_prob = Float(0.0)
    remesh_frequency = Int(5)
    cut_frequency = Int(0)
    min_hole_radius = Float(100.0)
    sigma_x = CStr('sigma_x')
    sigma_y = CStr('sigma_y')
    sigma_z = CStr('sigma_z')
    neck_theshold_low = Float(-1e-4)
    neck_threshold_high = Float(1e-2)
    neck_first_iter = Int(9)
    # method = Enum(DESCENT_METHODS)
    minimum_edge_length = Float(-1.0)

    def execute(self, namespace):
        import numpy as np
        from ch_shrinkwrap import _membrane_mesh as membrane_mesh

        mesh = membrane_mesh.MembraneMesh(mesh=namespace[self.input], 
                                        #   search_k=self.search_k,
                                          kc=self.kc,
                                          #kg=self.kg,
                                          max_iter=self.max_iters,
                                          step_size=self.curvature_weight,
                                          #skip_prob=self.skip_prob,
                                          remesh_frequency=self.remesh_frequency,
                                          delaunay_remesh_frequency=self.cut_frequency, # self.delaunay_remesh_frequency,
                                          delaunay_eps=self.min_hole_radius,
                                          neck_threshold_low = self.neck_theshold_low,
                                          neck_threshold_high = self.neck_threshold_high,
                                          neck_first_iter = self.neck_first_iter,
                                          shrink_weight = self.shrink_weight) # self.min_hole_radius)
                                          #a=self.attraction_weight,
                                          #c=self.curvature_weight,
                                        #   search_rad=self.search_rad)

        # try and close holes in the mesh before we start
        mesh.repair()
        mesh.remesh()

        namespace[self.output] = mesh

        im = namespace[self.input_image]
        weights = im.data_xyztc[:,:,:,0,0]

        vx, vy, vz = im.voxelsize_nm
        ox, oy, oz = im.origin

        x, y, z = np.mgrid[0:weights.shape[0], 0:weights.shape[1], 0:weights.shape[2]]

        x = ox + vx*x.ravel()
        y = oy + vy*y.ravel()
        z = oz + vz*z.ravel()

        weights=weights.ravel()
        mask = weights > 0
        weights = weights[mask]

        
        pts = np.ascontiguousarray(np.vstack([x[mask], 
                                              y[mask],
                                              z[mask]]).T)

        sigma = vx

        # from PYME.util import mProfile
        # mProfile.profileOn(['membrane_mesh.py'])
        mesh.shrink_wrap(pts, sigma=sigma, weights=np.repeat(weights, 3), method='conjugate_gradient', minimum_edge_length=self.minimum_edge_length)
        # mProfile.profileOff()
        # mProfile.report()

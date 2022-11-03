import logging
import time

from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, CStr, Int, Bool, Float, List

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
    punch_frequency = Int(0)
    min_hole_radius = Float(100.0)
    sigma_x = CStr('sigma_x')
    sigma_y = CStr('sigma_y')
    sigma_z = CStr('sigma_z')
    neck_threshold_low = Float(-1e-4)
    neck_threshold_high = Float(1e-2)
    neck_first_iter = Int(9)
    # method = Enum(DESCENT_METHODS)
    minimum_edge_length = Float(-1.0)

    def execute(self, namespace):
        import numpy as np
        from ch_shrinkwrap import _membrane_mesh as membrane_mesh
        from PYME.IO import MetaDataHandler

        inp = namespace[self.input]
        md = MetaDataHandler.DictMDHandler(getattr(inp, 'mdh', None)) # get metadata from the input dataset if present
        mesh = membrane_mesh.MembraneMesh(mesh=inp, 
                                        #   search_k=self.search_k,
                                          kc=self.kc,
                                          #kg=self.kg,
                                          max_iter=self.max_iters,
                                          step_size=self.curvature_weight,
                                          #skip_prob=self.skip_prob,
                                          remesh_frequency=self.remesh_frequency,
                                          delaunay_remesh_frequency=self.punch_frequency, # self.delaunay_remesh_frequency,
                                          delaunay_eps=self.min_hole_radius,
                                          neck_threshold_low = self.neck_threshold_low,
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
        start = time.time()
        mesh.shrink_wrap(pts, sigma, method='conjugate_gradient', minimum_edge_length=self.minimum_edge_length)
        stop = time.time()
        duration = stop-start
        md[f'Processing.ShrinkwrapMembrane.Runtime'] = duration
        # mProfile.profileOff()
        # mProfile.report()

        self._params_to_metadata(md)
        mesh.mdh = md

@register_module('ScreenedPoissonMesh')
class ScreenedPoissonMesh(ModuleBase):
    input = Input('filtered_localizations')
    output = Output('membrane')

    k = Int(10)
    smoothiter = Int(0)
    flipflag = Bool(False)
    viewpos = List([0,0,0])
    visiblelayer = Bool(False)
    depth = Int(8)
    fulldepth = Int(5)
    cgdepth = Int(0)
    scale = Float(1.1)
    samplespernode = Float(1.5)
    pointweight = Float(4)
    iters = Int(8)
    confidence = Bool(False)
    preclean = Bool(False)
    threads = Int(8)
    use_normals = Bool(False)

    def execute(self, namespace):
        import numpy as np
        from ch_shrinkwrap.screened_poisson import screened_poisson
        from ch_shrinkwrap import _membrane_mesh as membrane_mesh
        from PYME.IO import MetaDataHandler

        inp = namespace[self.input]
        md = MetaDataHandler.DictMDHandler(getattr(inp, 'mdh', None)) # get metadata from the input dataset if present
        points = np.ascontiguousarray(np.vstack([inp['x'], 
                                                 inp['y'],
                                                 inp['z']]).T)
        
        if self.use_normals:
            try:
                normals = np.ascontiguousarray(np.vstack([inp['xn'], 
                                                        inp['yn'],
                                                        inp['zn']]).T)
            except KeyError:
                normals = None
        else:
            normals = None

        start = time.time()
        vertices, faces = screened_poisson(points, normals, k=self.k, 
                                           smoothiter=self.smoothiter, flipflag=self.flipflag,
                                           viewpos=np.array(self.viewpos), visiblelayer=self.visiblelayer,
                                           depth=self.depth, fulldepth=self.fulldepth, cgdepth=self.cgdepth, 
                                           scale=self.scale, samplespernode=self.samplespernode, 
                                           pointweight=self.pointweight, iters=self.iters, 
                                           confidence=self.confidence, preclean=self.preclean,
                                           threads=self.threads)
        stop = time.time()
        duration = stop-start
        md[f'Processing.ScreenedPoissonMesh.Runtime'] = duration
        self._params_to_metadata(md)

        mesh = membrane_mesh.MembraneMesh(vertices=vertices, faces=faces)

        mesh.mdh = md

        namespace[self.output] = mesh

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
    neck_threshold_low = Float(-1e-4)
    neck_threshold_high = Float(1e-2)
    neck_first_iter = Int(9)
    # method = Enum(DESCENT_METHODS)
    minimum_edge_length = Float(-1.0)

    def execute(self, namespace):
        import numpy as np
        from ch_shrinkwrap import _membrane_mesh as membrane_mesh
        from PYME.IO import MetaDataHandler

        inp = namespace[self.input]

        mesh = membrane_mesh.MembraneMesh(mesh=inp, 
                                        #   search_k=self.search_k,
                                          kc=self.kc,
                                          #kg=self.kg,
                                          max_iter=self.max_iters,
                                          step_size=self.curvature_weight,
                                          #skip_prob=self.skip_prob,
                                          remesh_frequency=self.remesh_frequency,
                                          delaunay_remesh_frequency=self.cut_frequency, # self.delaunay_remesh_frequency,
                                          delaunay_eps=self.min_hole_radius,
                                          neck_threshold_low = self.neck_threshold_low,
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

        md = MetaDataHandler.DictMDHandler(getattr(inp, 'mdh', None)) # get metadata from the input dataset if present
        self._params_to_metadata(md)
        mesh.mdh = md

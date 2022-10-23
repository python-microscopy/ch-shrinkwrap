import logging

import numpy as np

from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, CStr, Float, Bool, Int, List
from PYME.IO import tabular

logger = logging.getLogger(__name__)

@register_module('PointcloudFromShape')
class PointcloudFromShape(ModuleBase):
    output = Output('two_toruses')

    shape_name = CStr('TwoToruses')
    shape_params = CStr("{'r': 30, 'R': 100}")
    density = Float(1.0)
    p = Float(0.01)
    psf_width_x = Float(280.0)
    psf_width_y = Float(280.0)
    psf_width_z = Float(840.0)
    mean_photon_count = Int(600)
    bg_photon_count = Int(20)
    noise_fraction = Float(0.1)
    no_jitter = Bool(False)

    def execute(self, namespace):
        from numpy import sqrt 
        import yaml
        from ch_shrinkwrap.evaluation_utils import generate_smlm_pointcloud_from_shape
        from PYME.IO.tabular import ColumnSource
        from PYME.IO import MetaDataHandler
        
        params = yaml.load(self.shape_params, Loader=yaml.FullLoader)
        if self.no_jitter:
            psf_width = None
        else:
            psf_width = psf_width=(self.psf_width_x, self.psf_width_y, self.psf_width_z)
        points, normals, sigma = generate_smlm_pointcloud_from_shape(self.shape_name, params, 
                                                                     density=self.density, p=self.p, 
                                                                     psf_width=psf_width, 
                                                                     mean_photon_count=self.mean_photon_count, 
                                                                     bg_photon_count=self.bg_photon_count, 
                                                                     noise_fraction=self.noise_fraction)

        if self.no_jitter:
            ds = ColumnSource(x=points[:,0], y=points[:,1], z=points[:,2],
                              xn=normals[:,0], yn=normals[:,1], zn=normals[:,2])
        else:
            s = sqrt((sigma*sigma).sum(1))
            
            ds = ColumnSource(x=points[:,0], y=points[:,1], z=points[:,2], 
                              xn=normals[:,0], yn=normals[:,1], zn=normals[:,2],
                              sigma=s, sigma_x=sigma[:,0], sigma_y=sigma[:,1], 
                              sigma_z=sigma[:,2])
                              
        md = MetaDataHandler.DictMDHandler()
        self._params_to_metadata(md)
        ds.mdh = md

        namespace[self.output] = ds

@register_module('AddAllMetadataToPipeline')         
class AddAllMetadataToPipeline(ModuleBase):
    """Copies AddMetadataToMeasurements but with a twist
    """
    inputMeasurements = Input('measurements')
    outputName = Output('annotatedMeasurements')
    
    def execute(self, namespace):
        res = {}
        meas = namespace[self.inputMeasurements]
        res.update(meas)
        
        nEntries = len(list(res.values())[0])
        for k in meas.mdh.keys():
            v = meas.mdh[k]
            if isinstance(v, List) or isinstance(v, list):
                v = str(v)
            res[k] = np.array([v]*nEntries)
        
        res = tabular.MappingFilter(res)
        
        namespace[self.outputName] = res

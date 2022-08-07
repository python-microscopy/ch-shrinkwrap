from PYME.recipes.base import register_module, ModuleBase
from PYME.recipes.traits import Input, Output, CStr, Float, ListFloat
import logging

logger = logging.getLogger(__name__)

@register_module('PointcloudFromShape')
class PointcloudFromShape(ModuleBase):
    input = Input('filtered_localizations')
    output = Output('two_toruses')

    shape_name = CStr('TwoToruses')
    shape_params = CStr("{'r': 30, 'R': 100}")
    density = Float(1.0)
    p = Float(0.01)
    psf_width_x = Float(280.0)
    psf_width_y = Float(280.0)
    psf_width_z = Float(840.0)
    mean_photon_count = Float(600.0)
    bg_photon_count = Float(20.0)
    noise_fraction = Float(0.1)

    def execute(self, namespace):
        from numpy import sqrt 
        import yaml
        from ch_shrinkwrap.evaluation_utils import generate_smlm_pointcloud_from_shape
        from PYME.IO.tabular import ColumnSource
        
        params = yaml.load(self.shape_params, Loader=yaml.FullLoader)
        points, sigma = generate_smlm_pointcloud_from_shape(self.shape_name, params, 
                                                            density=self.density, p=self.p, 
                                                            psf_width=(self.psf_width_x, 
                                                            self.psf_width_y, self.psf_width_z), 
                                                            mean_photon_count=self.mean_photon_count, 
                                                            bg_photon_count=self.bg_photon_count, 
                                                            noise_fraction=self.noise_fraction)

        s = sqrt((sigma*sigma).sum(1))
        
        ds = ColumnSource(x=points[:,0], y=points[:,1], z=points[:,2], 
                          sigma=s, sigma_x=sigma[:,0], sigma_y=sigma[:,1], 
                          sigma_z=sigma[:,2])

        namespace[self.output] = ds

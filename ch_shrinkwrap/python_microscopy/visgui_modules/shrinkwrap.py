import numpy as np

def shrinkwrap(visFr):
    from PYME.LMVis.Extras.extra_layers import gen_isosurface
    from PYME.LMVis.layers.mesh import TriangleRenderLayer
    from ch_shrinkwrap.python_microscopy.recipe_modules.surface_fitting import ShrinkwrapMembrane

    if not 'surf0' in visFr.pipeline.dataSources.keys():
        gen_isosurface(visFr)

    membrane_name = visFr.pipeline.new_ds_name('membrane')
    
    recipe = visFr.pipeline.recipe
    sw = ShrinkwrapMembrane(recipe, invalidate_parent=False, input='surf0', output=membrane_name, points='filtered_localizations')
    
    if sw.configure_traits(kind='modal'):
        recipe.add_module(sw)
        recipe.execute()
        surf_count = 0
        layer = TriangleRenderLayer(visFr.pipeline, dsname=membrane_name, method='shaded', cmap = ['C', 'M', 'Y', 'R', 'G', 'B'][surf_count % 6])
        visFr.add_layer(layer)
        sw._invalidate_parent = True

        visFr.RefreshView()

def Plug(visFr):
    visFr.AddMenuItem('Mesh', 'Shrinkwrap membrane surface', lambda e: shrinkwrap(visFr))
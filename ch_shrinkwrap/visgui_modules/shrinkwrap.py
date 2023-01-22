import numpy as np

def shrinkwrap(visFr):
    from PYME.LMVis.Extras.extra_layers import gen_isosurface
    from PYME.LMVis.layers.mesh import TriangleRenderLayer
    from ch_shrinkwrap.recipe_modules.surface_fitting import ShrinkwrapMembrane

    surf_name = 'surf0'

    if not surf_name in visFr.pipeline.dataSources.keys():
        gen_isosurface(visFr)

    membrane_name = visFr.pipeline.new_ds_name('membrane')
    
    recipe = visFr.pipeline.recipe
    sw = ShrinkwrapMembrane(recipe, invalidate_parent=False, input=surf_name, output=membrane_name, input_points='filtered_localizations')
    
    if sw.configure_traits(kind='modal'):
        recipe.add_module(sw)
        recipe.execute()
        surf_count = 0
        layer = TriangleRenderLayer(visFr.pipeline, dsname=membrane_name, method='shaded')
        visFr.add_layer(layer)
        sw._invalidate_parent = True

        visFr.RefreshView()

def Plug(visFr):
    visFr.AddMenuItem('Mesh', 'Shrinkwrap membrane surface', lambda e: shrinkwrap(visFr))
    
import numpy as np

def gen_membrane(visFr):
    from PYME.LMVis.layers.triangle_mesh import TriangleRenderLayer
    from ch_shrinkwrap.python_microscopy.recipe_modules.surface_fitting import MembraneDualMarchingCubes
    
    if not 'octree' in visFr.pipeline.dataSources.keys():
        from PYME.LMVis.Extras import extra_layers
        extra_layers.gen_octree_layer_from_points(visFr)

    
    recipe = visFr.pipeline.recipe
    dmc = MembraneDualMarchingCubes(recipe, invalidate_parent=False, 
                                            input='octree',
                                             output='membrane')
    if dmc.configure_traits(kind='modal'):
        recipe.add_module(dmc)
        recipe.execute()

        layer = TriangleRenderLayer(visFr.pipeline, dsname='membrane')
        visFr.add_layer(layer)
        dmc._invalidate_parent = True

def shrinkwrap(visFr):
    from ch_shrinkwrap.python_microscopy.recipe_modules.surface_fitting import ShrinkwrapMembrane

    if not 'membrane' in visFr.pipeline.dataSources.keys():
        gen_membrane(visFr)
    
    recipe = visFr.pipeline.recipe
    sw = ShrinkwrapMembrane(recipe, invalidate_parent=False, input='membrane', points='filtered_localizations')
    
    if sw.configure_traits(kind='modal'):
        recipe.add_module(sw)
        recipe.execute()
        sw._invalidate_parent = True

        visFr.RefreshView()

def Plug(visFr):
    visFr.AddMenuItem('View', 'Generate Membrane Surface', lambda e: gen_membrane(visFr))
    visFr.AddMenuItem('View', 'Shrinkwrap Membrane Surface', lambda e: shrinkwrap(visFr))
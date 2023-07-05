import numpy as np

def skeletonize(visFr):
    from PYME.LMVis.layers.mesh import TriangleRenderLayer
    from ch_shrinkwrap.recipe_modules.surface_feature_extraction import SkeletonizeMembrane
    
    surf_name = 'surf0'
    skeleton_name = visFr.pipeline.new_ds_name('skeleton')

    recipe = visFr.pipeline.recipe
    sw = SkeletonizeMembrane(recipe, invalidate_parent=False, input=surf_name, output=skeleton_name)
    
    if sw.configure_traits(kind='modal'):
        recipe.add_module(sw)
        recipe.execute()
        surf_count = 0
        layer = TriangleRenderLayer(visFr.pipeline, dsname=skeleton_name, method='shaded')
        visFr.add_layer(layer)
        sw._invalidate_parent = True

        visFr.RefreshView()

def Plug(visFr):
    visFr.AddMenuItem('Mesh', 'Skeletonize mesh', lambda e: skeletonize(visFr))
    

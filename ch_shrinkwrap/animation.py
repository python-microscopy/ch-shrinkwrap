import os
from PIL import Image

def animate_shrinkwrap(mesh, pts, sigma, layer, pymevis, save_dir):
    """Create an animation of a shrinkwrapping event.

    Parameters
    ----------
    mesh : ch_shrinkwrap._membrane_mesh.MembraneMesh
        membrane mesh
    pts : np.array
        (N,3) array of points to shrinkwrap to
    sigma : np.array
        (N,) array of point uncertainty
    layer : PYME.LMVis.layers.mesh.TriangleRenderLayer
        GUI layer where mesh is stored
    pymevis : PYME.LMVis.VisGUI.VisGUIFrame
        The PYMEVisualise frame containing the layer
    save_dir : str
        Folder where we want to save the animation
    """

    # Grab the number of iterations and force single iterations
    max_iters = mesh.max_iter
    mesh.max_iter = 1

    # Make a save directory, if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Iterate 
    for _i in range(max_iters):
        # Shrink wrap
        mesh.shrink_wrap(pts, sigma)
        
        # Force a layer update for visualisation 
        layer.update()
        
        # Assumes PYME.LMVis.gl_render3D_shaders
        snap = pymevis.glCanvas.getIm().transpose(1,0,2)
        
        # Save the image
        Image.fromarray(snap).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_dir, 'frame{:04d}.{}'.format(_i,'png')))
    
    mesh.max_iter = max_iters
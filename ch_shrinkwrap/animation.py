import os
import numpy as np
from PIL import Image

def animate_shrinkwrap(mesh, pts, sigma, layer, pymevis, save_dir):
    """Create an animation of a shrinkwrapping event

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

    # Steal parameters
    max_iters = mesh.max_iter
    mesh.max_iter = 1
    delaunay_remesh_frequency = mesh.delaunay_remesh_frequency
    mesh.delaunay_remesh_frequency = 0
    remesh_frequency = mesh.remesh_frequency
    mesh.remesh_frequency = 0
    dr = (delaunay_remesh_frequency != 0)
    r = (remesh_frequency != 0)
    if r:
        initial_length = mesh._mean_edge_length
        final_length = np.max(sigma)
        m = (final_length - initial_length)/max_iters

    # Make a save directory, if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Iterate 
    for _i in range(max_iters):
        # Shrink wrap
        mesh.shrink_wrap(pts, sigma)

        # Remesh
        if (_i != 0) and r and ((_i % remesh_frequency) == 0):
            target_length = initial_length + m*_i
            mesh.remesh(5, target_length, 0.5, 10)
            print('Target mean length: {}   Resulting mean length: {}'.format(str(target_length), 
                                                                            str(mesh._mean_edge_length)))

        # Delaunay remesh
        if (_i != 0) and dr and ((_i % delaunay_remesh_frequency) == 0):
            mesh.delaunay_remesh(pts)
        
        # Force a layer update for visualisation 
        layer.update()
        
        # Assumes PYME.LMVis.gl_render3D_shaders
        snap = pymevis.glCanvas.getIm().transpose(1,0,2)
        
        # Save the image
        Image.fromarray(snap).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_dir, 'frame{:04d}.{}'.format(_i,'png')))
    
    # Restore
    mesh.max_iter = max_iters
    mesh.remesh_frequency = remesh_frequency
    mesh.delaunay_remesh_frequency = delaunay_remesh_frequency
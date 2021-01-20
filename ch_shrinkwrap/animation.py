import os
import numpy as np
from PIL import Image

def animate_shrinkwrap(mesh, pts, sigma, layer, pymevis, save_dir, return_curvature_mean_hists=False):
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
    return_curvature_mean_hists : bool
        Boolean to store curvature histogram means
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
        final_length = 4.5 # 3*np.max(sigma)
        m = (final_length - initial_length)/max_iters

    # Make a save directory, if needed
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if return_curvature_mean_hists:
        edges = np.linspace(-0.02,0.02,100)
        hists = np.zeros((max_iters,len(edges)-1))
        hists[0,:], _ = np.histogram(mesh.curvature_mean[mesh._vertices['halfedge']!=-1],bins=edges,density=True)
        means = np.zeros(max_iters)

    # Grab the original
    #Force a layer update for visualisation 
    layer.update()
    
    # Assumes PYME.LMVis.gl_render3D_shaders
    snap = pymevis.glCanvas.getIm().transpose(1,0,2)
    
    # Save the image
    Image.fromarray(snap).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(save_dir, 'frame{:04d}.{}'.format(0,'png')))

    # Iterate 
    for _i in range(1,max_iters):
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
            mesh.delaunay_remesh(pts, sigma)

        if return_curvature_mean_hists:
            hists[_i,:], _ = np.histogram(mesh.curvature_mean[mesh._vertices['halfedge']!=-1],bins=edges,density=True)
            means[_i] = np.mean(mesh.curvature_mean[mesh._vertices['halfedge']!=-1])
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

    if return_curvature_mean_hists:
        return hists, edges, means
    return 0, 0, 0
# Canham-Helfrich Shrinkwrap

Fit a surface through single-molecule localization microscopy data, taking 
localization precision into account, and constraining the fit on curvature 
likelihood. Curvature likelihood is based on the Canham-Helfrich energy
functional, with stiffness coefficients of lipid bilayers extracted from 
the literature.

## Requirements

- python-microscopy
- pymeshlab

## Installation

ch_shrinkwrap is a plugin for the [PYthon Microscopy Environment](https://python-microscopy.org/). 
Once PYME is installed, open a command line and execute the following. 

0. Clone this repository.

1. Navigate to the cloned folder `ch-shrinkwrap`.

1. `python setup.py install`

The plugin will appear in PYMEVisualize (`pymevis`) under *Mesh-->Shrinkwrap membrane surface*.

## Usage

This is designed to fit coarse isosurfaces to localization data.

1. Create an isosurface that slightly over-approximates your data by going to *Mesh-->Generate isosurface*.
2. Navigate to *Mesh-->Shrinkwrap membrane surface*. A dialog will pop up. Parameters for this dialog are described below.

| Parameter                  | Description                                                                                                                                                                                        | Standard values        |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| input                      | The data source containing the coarse isosurface.                                                                                                                                                  | surf0                  |
| curvature_weight           | The contribution of curvature (vs. point attraction force) to the fitting procedure. Higher values create smoother surfaces.                                                                       | 10 - 100               |
| kc                         | Lipid stiffness coefficient of membrane in eV (can be looked up in the literature).                                                                                                                | 0 - 1 (20k_bT)         |
| max_iters                  | Maximum number of fitting iterations.                                                                                                                                                              | 10 - 100               |
| points                     | The data source containing the points to fit.                                                                                                                                                      | filtered_localizations |                     
| remesh_frequency           | Remesh the isourface every N iterations. Helps keep the fitting numerically stable. Should be often.                                                                                               | 5                      |
| punch_frequency           | Every punch_frequency iterations, check for and add holes in regions of the mesh where there is a continuous empty area in between two "sides" of the mesh.                                                                                               | 0                      |
| min_hole_radius           | The minimum size of the continuous empty area defined in punch_frequency.                                                                                               | 100.0                      |
| neck_first_iter           | Every neck_first_iter iterations, check for and remove necks in the mesh.                                                                                               | 9                      |
| neck_threshold_low           | Vertices with Gaussian curvature below this threshold are necks.                                                                                               | -1e4                      |
| neck_threshold_high           | Vertices with Gaussian curvature above this threshold are necks.                                                                                               | 1e2                      |
| sigma_x                    | The variable in the `points` data source containing localization precision in the x-direction. If sigma is only known for one direction, supply it here and it will be assumed for all directions. | error_x                |
| sigma_y                    | The variable in the `points` data source containing localization precision in the y-direction.                                                                                                     |                        |
| sigma_z                    | The variable in the `points` data source containing localization precision in the z-direction.                                                                                                     |                        |
| output                     | The name of the data source that will contain the fit isosurface.                                                                                                                                  | membrane0              |

NOTE: If `points` contains multiple channels (multicolor data), points in all channels will be fit as a single point set.

# Test suite configuration

The test suite will compare a mesh generated from a noisy approximation of a shape
to the shape's actual structure. The test suite runs as a recipe pipeline
(see https://python-microscopy.org/doc/pymevis/edit_recipe.html) within PYME.

An example configuration (YAML) file for the test suite can be found at ch_shrinkwrap/test_example.yaml.

An example recipe file for Shrinkwrapping can be found at ch_shrinkwrap/test_evaluation_recipe.yaml.

## Parameters

This section lists the parameters included in a test suite configuration file.

- save_fp : str
    Path indicating where to save test suite intermediate files.
    
### System

The parameters are related to the behaviour of the optical system used to generate the 
theoretically simulated images that were localized to create the point cloud. In reality, 
we go directly to the point cloud. See ch_shrinkwrap.recipe_modules.generate.PointcloudFromShape.

- psf_width_x : float
    The FWHM (2.355 x sigma) of the PSF along the x-dimension.
- psf_width_y : float
    The FWHM (2.355 x sigma) of the PSF along the y-dimension.
- psf_width_z : float
    The FWHM (2.355 x sigma) of the PSF along the z-dimension.
- mean_photon_count : int
    Average number of photons within a PSF.
- bg_photon_count : int
    Average number of photons in an empty pixel.
    
### Shape

These are related to the theoretical structure, defined by a signed-distance function
(see ch_shrinkwrap.shape), that gives rise to the simulated point cloud. See
ch_shrinkwrap.recipe_modules.generate.PointcloudFromShape.

- type : str
    Name of the shape. The class name of an object derived from ch_shrinkwrap.shape.Shape,
    and located in ch_shrinkwrap.shape.
- parameters : dict
    The arguments passed to `__init__` for a shape of class type. 

### Point cloud {#cloud}

The parameters define the sampling of the point cloud and the amount of background noise 
added to the point cloud. See ch_shrinkwrap.recipe_modules.generate.PointcloudFromShape.

- density : float
    The density of the point cloud. Sampled on a regular grid, this is 1/(dx^3) where
    dx is the grid spacing.
- p : float
    The "acceptance probability" of a randomly generated point in the point cloud. This
    further decreases the density.
- noise_fraction : float
    The amount of background noise in the point cloud, as a fraction between 0 and 1. At
    0.1, 10 percent of the localizations in the point cloud are background noise.

### Dual marching cubes

These are the parameters used to generate the initial isosurface. By default we assume
smooth_curvature, repair, remesh, and cull_inner_surfaces are all true. See 
PYME.experimental.recipes.surface_fitting_DualMarchingCubes.

Note that, unlike the rest of the parameters in this file, which vary independently of
one another, `threshold_density` and `n_points_min` vary together along with 
[Point Cloud](#cloud)'s `p` and `density`.

- threshold_density : float
    The point cloud density at which a surface divides signal from background with a
    surface.
- n_points_min : int
    The minimum number of points required in a volume to estimate point cloud density.
    
### Shrinkwrapping

These are the parameters used to shrinkwrap the initial mesh generated by Dual Marching Cubes
to the simulated points. See ch_shrinkwrap.recipe_modules.surface_fitting.ShrinkwrapMembrane.

The parameters kc, sigma_x, sigma_y, sigma_z and minimum_edge_length are left at sensible
defaults.

- max_iters : int
    Maximum number of fitting iterations to run.
- curvature_weight : float
    The influence of the curvature force relative to the point attraction force.
- remesh_frequency : int
    Remesh (improve numerical quality) the mesh every remesh_frequency iterations.
- punch_frequency : int
    Every punch_frequency iterations, check for and add holes in regions of the mesh 
    where there is a continuous empty area in between two "sides" of the mesh.
- min_hole_radius : float
    The minimum size of the continuous empty area.
- neck_first_iter : int
    Every neck_first_iter iterations, check for and remove necks in the mesh.
- neck_threshold_low : float
    Vertices with Gaussian curvature below this threshold are necks.
- neck_threshold_high : float
    Vertices with Gaussian curvature above this threshold are necks.

### Screened poisson reconstruction 

These are the parameters used to generated a Screened Poisson Reconstruction (SPR) surface from 
a point cloud. SPR is a state-of-the-art surface reconstruction algorithm for point clouds,
designed for use with range scanning and LIDAR. See 
ch_shrinkwrap.recipe_modules.surface_fitting.ScreenedPoissonMesh. 

Note that he first few parameters in ScreenedPoissonMesh are related to normal estimation from 
a point cloud. These are ignored in the evaluation pipeline as we pass normals calculated directly
from the signed distance function of the test shape to SPR for reconstruction.

The parameters visiblelayer, depth, fulldepth, cgdepth, scale, confidence, preclean and threads are 
left at sensible defaults (see ch_shrinkwrap.screened_poisson.screened_poisson).

- samplespernode : float
     The minimum number of sample points that should fall within an octree node as the octree
     construction is adapted to sampling density. For noise-free samples, small values in the 
     range [1.0 - 5.0] can be used. For more noisy samples, larger values in the range 
     [15.0 - 20.0] may be needed to provide a smoother, noise-reduced, reconstruction.
- pointweight : float
    This floating point value specifies the importants that interpolation of the point samples 
    is given in the formulation of the screened Poisson equation. The results of the original
    (unscreened) Poisson Reconstruction can be obtained by setting this value to 0. The default 
    value for this parameter is 4.
- iters : int
    Gauss-Seidel Relaxations: This integer value specifies the number of Gauss-Seidel relaxations 
    to be performed at each level of the hierarchy. The default value for this parameter is 8.

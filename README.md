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
| curvature weight           | The contribution of curvature (vs. point attraction force) to the fitting procedure. Higher values create smoother surfaces.                                                                       | 10 - 100               |
| kc                         | Lipid stiffness coefficient of membrane in eV (can be looked up in the literature).                                                                                                                | 0 - 1 (20k_bT)         |
| max iters                  | Maximum number of fitting iterations.                                                                                                                                                              | 10 - 100               |
| points                     | The data source containing the points to fit.                                                                                                                                                      | filtered_localizations |                     
| remesh frequency           | Remesh the isourface every N iterations. Helps keep the fitting numerically stable. Should be often.                                                                                               | 5                      |
| sigma x                    | The variable in the `points` data source containing localization precision in the x-direction. If sigma is only known for one direction, supply it here and it will be assumed for all directions. | error_x                |
| sigma y                    | The variable in the `points` data source containing localization precision in the y-direction.                                                                                                     |                        |
| sigma z                    | The variable in the `points` data source containing localization precision in the z-direction.                                                                                                     |                        |
| output                     | The name of the data source that will contain the fit isosurface.                                                                                                                                  | membrane0              |

NOTE: If `points` contains multiple channels (multicolor data), points in all channels will be fit as a single point set.
 
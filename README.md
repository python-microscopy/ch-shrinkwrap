# ch_shrinkwrap

Fit a surface through single-molecule localization microscopy data, taking 
localization precision into account, and constraining the fit on curvature 
likelihood. Curvature likelihood is based on the Canham-Helfrich energy
functional, with stiffness coefficients of lipid bilayers extracted from 
the literature.

## Installation

ch_shrinkwrap is a plugin for the [PYthon Microscopy Environment](https://python-microscopy.org/). 
Once PYME is installed, open a command line and execute the following. 

0. Clone this repository and navigate to its root folder on your machine.

1. `python ch_shrinkwrap/setup.py install`

2. `python ch_shrinkwrap/install_plugin.py`

The plugin will appear in PYMEVisualize (`pymevis`) under "Mesh-->Shrinkwrap membrane surface".

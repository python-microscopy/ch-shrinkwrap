# Canham-Helfrich Shrinkwrap

Fit a surface through single-molecule localization microscopy data, taking 
localization precision into account, and constraining the fit on curvature 
likelihood. Curvature likelihood is based on the Canham-Helfrich energy
functional, with stiffness coefficients of lipid bilayers extracted from 
the literature.

## Installation

ch_shrinkwrap is a plugin for the [PYthon Microscopy Environment](https://python-microscopy.org/). 
Once PYME is installed, open a command line and execute the following. 

0. Clone this repository.

1. `python ch-shrinkwrap/setup.py install`

2. `python ch-shrinkwrap/install_plugin.py`

The plugin will appear in PYMEVisualize (`pymevis`) under *Mesh-->Shrinkwrap membrane surface*.

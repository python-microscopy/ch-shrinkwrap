#!/usr/bin/env python

import os
import numpy
import PYME
import PYME.experimental
from setuptools import setup, Extension
from Cython.Build import cythonize

this_dir = os.path.dirname(os.path.abspath(__file__))
ch_dir = os.path.join(this_dir, 'ch_shrinkwrap')
pyme_exp_dir = os.path.dirname(PYME.experimental.__file__)
# parent of the PYME package dir — where Cython searches for cimported .pxd files
pyme_pkg_root = os.path.dirname(os.path.dirname(PYME.__file__))

cython_include_dirs = [numpy.get_include(), ch_dir, pyme_exp_dir]

module0 = Extension('ch_shrinkwrap.membrane_mesh_utils',
                    sources=[os.path.join(ch_dir, 'membrane_mesh_utils.c')])
module1 = Extension('ch_shrinkwrap._membrane_mesh',
                    sources=[os.path.join(ch_dir, '_membrane_mesh.pyx')],
                    include_dirs=cython_include_dirs)
module2 = Extension('ch_shrinkwrap._skeleton_mesh',
                    sources=[os.path.join(ch_dir, '_skeleton_mesh.pyx')],
                    include_dirs=cython_include_dirs)
module3 = Extension('ch_shrinkwrap.conj_grad_utils',
                    sources=[os.path.join(ch_dir, 'conj_grad_utils.c')])

setup(ext_modules=cythonize([module0, module1, module2, module3],
                            include_path=[pyme_pkg_root],
                            language_level=3))

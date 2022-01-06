#!/usr/bin/env python

import os
from setuptools import find_packages, Extension
from numpy.distutils.core import setup
from numpy.distutils.misc_util import get_numpy_include_dirs
from Cython.Build import cythonize

this_dir = os.path.dirname(__file__)

module0 = Extension('ch_shrinkwrap.membrane_mesh_utils', sources=[os.path.join(this_dir, 'ch_shrinkwrap/membrane_mesh_utils.c')])
module1 = Extension('ch_shrinkwrap._membrane_mesh', sources=[os.path.join(this_dir, 'ch_shrinkwrap/_membrane_mesh.pyx')], include_dirs=get_numpy_include_dirs())
#module2 = Extension('ch_shrinkwrap._conj_grad', sources=[os.path.join(this_dir, 'ch_shrinkwrap/_conj_grad.pyx')], include_dirs=get_numpy_include_dirs())

setup(name='ch_shrinkwrap',
      version='0.0',
      package_path=os.path.join(this_dir,'ch_shrinkwrap'),
      description='Mesh shrinkwrapping constrained by Canham-Helfrich energy functional.',
      author='Zach Marin',
      author_email='zach.marin@yale.edu',
      packages=find_packages(),
      ext_modules = cythonize([module0, module1]) #, module2])
     )
#!/usr/bin/env python

from setuptools import find_packages, Extension
from numpy.distutils.core import setup
from numpy.distutils.misc_util import get_numpy_include_dirs
from Cython.Build import cythonize

module0 = Extension('ch_shrinkwrap.membrane_mesh_utils', sources=['ch_shrinkwrap/membrane_mesh_utils.c'])
module1 = Extension('ch_shrinkwrap.sdf_octree.csdf_octree', sources=['ch_shrinkwrap/sdf_octree/csdf_octree.pyx'], include_dirs=get_numpy_include_dirs())

setup(name='ch_shrinkwrap',
      version='0.0',
      description='Mesh shrinkwrapping constrained by Canham-Helfrich energy functional.',
      author='Zach Marin',
      author_email='zach.marin@yale.edu',
      packages=find_packages(),
      ext_modules = cythonize([module0, module1])
     )
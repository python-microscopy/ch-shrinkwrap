#!/usr/bin/env python

from setuptools import find_packages, Extension
from numpy.distutils.core import setup

module1 = Extension('membrane_mesh_utils', sources = ['ch_shrinkwrap/membrane_mesh_utils.c'])

setup(name='ch_shrinkwrap',
      version='0.0',
      description='Mesh shrinkwrapping constrained by Canham-Helfrich energy functional.',
      author='Zach Marin',
      author_email='zach.marin@yale.edu',
      packages=find_packages(),
      ext_modules = [module1]
     )
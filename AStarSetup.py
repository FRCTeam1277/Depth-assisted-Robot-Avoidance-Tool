""" Converts AStar algorithm into c code to make it way more efficent
Use `python setup.py build_ext --inplace` to run (DON'T RUN DIRECTLY)
Run this on the raspberry pi to have it built properly!
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("AStar.pyx")
)
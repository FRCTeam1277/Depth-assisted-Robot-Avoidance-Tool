""" Converts AStar algorithm into c code to make it way more efficent
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("AStar.pyx")
)
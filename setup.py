from setuptools import setup, find_namespace_packages
import os

here = os.path.dirname(os.path.abspath(__file__))
pkgs = find_namespace_packages(where=here, include=['*'])
exclude = {'tests', 'tests.*', 'Labs', 'Labs.*', 'examples', 'examples.*'}
pkgs = [p for p in pkgs if not any(p == e or p.startswith(e + '.') for e in exclude)]
pkgs.insert(0, 'numfast')

setup(
    packages=pkgs,
    package_dir={'numfast': '.'},
)

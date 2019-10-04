import os
import sys
from setuptools import setup, find_packages
PACKAGE_NAME = 'hamiltorch'
MINIMUM_PYTHON_VERSION = 3, 5


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert 0, "'{0}' not found in '{1}'".format(key, module_path)


check_python_version()
setup(
    name='hamiltorch',
    version=read_package_variable('__version__'),
    description='A PyTorch-based library for Riemannian Manifold Hamiltonian Monte Carlo (RMHMC)',
    author='Adam D. Cobb, Atilim Gunes Baydin',
    author_email='acobb@robots.ox.ac.uk',
    packages=find_packages(),
    install_requires=['torch>=1.1.0', 'numpy', 'termcolor'],
    url='https://github.com/AdamCobb/hamiltorch',
    classifiers=['Development Status :: 4 - Beta', 'License :: OSI Approved :: BSD License', 'Programming Language :: Python :: 3.5'],
    license='BSD',
    keywords='pytorch riemannian hamiltonian monte carlo',
)

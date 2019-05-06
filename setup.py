#!/usr/bin/env python
#cython: language_level=3

import os
from setuptools import setup, Extension
import sys

from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import numpy.distutils

extra_link_args = ['-lblas']
include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()

extensions = cythonize(
    Extension(
        "nnattack.attacks.nns.cutils",
        ["nnattack/attacks/nns/cutils.pyx"],
        extra_link_args=extra_link_args,
        extra_compile_args=['-std=c11'],
        include_dirs=include_dirs,
    ),
)
cmdclasses = {'build_ext': build_ext}
setup_requires = [
]
install_requires = [
    'numpy',
    'scipy',
    'scikit-learn',
    'Cython',
    'joblib',
    'six',
    'cvxopt',
]
tests_require = [
    'nose',
    'coverage',
]


setup(
    name='attacks',
    version='0.0.1',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='',
    author_email='',
    url='',
    cmdclass=cmdclasses,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    test_suite='nnattack',
    packages=[
    ],
    package_dir={
    },
    ext_modules=extensions,
)

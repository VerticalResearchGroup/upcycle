import sys

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


setup(
    name='upcycle',
    version='0.1.0',
    author='Michael Davies',
    author_email='davies@cs.wisc.edu',
    packages=['upcycle', 'upcycle.model', 'upcycle.ops'],
    url='',
    license='LICENSE',
    description='An awesome package that does something',
    long_description=open('README.md').read(),
    install_requires=[],
    ext_modules=[
        Pybind11Extension('upcycle.model.cache', ['upcycle/model/cache.cc']),
        Pybind11Extension('upcycle.model.destlist', ['upcycle/model/destlist.cc']),
    ],
    zip_safe=False,
    python_requires=">=3.6",
)

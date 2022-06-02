import sys

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


setup(
    name='upcycle',
    version='0.1.0',
    author='Michael Davies',
    author_email='davies@cs.wisc.edu',
    packages=['upcycle', 'upcycle.model'],
    url='',
    license='LICENSE',
    description='An awesome package that does something',
    long_description=open('README.md').read(),
    install_requires=[],
    ext_modules=[
        Pybind11Extension(
            'upcycle.model.cache',
            ['upcycle/model/cache.cc'],
            # Example: passing in the version to the compiled code
            # define_macros = [('VERSION_INFO', __version__)],
        ),
    ],
    zip_safe=False,
    python_requires=">=3.6",
)

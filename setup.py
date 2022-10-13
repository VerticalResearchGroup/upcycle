import sys
import glob

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# class CustomPybindExtension(Pybind11Extension):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)





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
    install_requires=['pyyaml'],
    ext_modules=[
        Pybind11Extension(
            'upcycle.model.c_model',
            glob.glob("upcycle/model/*.cc"),
            depends=glob.glob("upcycle/model/*.hh"),
            undef_macros=[]),
    ],
    zip_safe=False,
    python_requires=">=3.6",
)

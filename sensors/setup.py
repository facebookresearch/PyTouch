# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import codecs
import os
import pathlib
import re
import shutil
import platform

import pkg_resources
from setuptools import find_packages, setup, Extension

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(BASE_DIR, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="pytouch_sensors",
    version=find_version("pytouch_sensors", "__init__.py"),
    description="PyTouch sensors library for interfacing with tactile sensors.",
    url="https://github.com/facebookresearch/pytouch/sensors",
    author="PyTouch",
    keywords=["science"],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="LICENSE",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "pyudev",
    ],
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)

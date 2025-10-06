#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="gt4py_ice_adjust",
    version="0.1.0",
    description="GT4Py translation of ice_adjust.F90 and dependencies from PHYEX",
    author="GT4Py Translation",
    packages=find_packages(),
    install_requires=[
        "gt4py>=1.0.1",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
)

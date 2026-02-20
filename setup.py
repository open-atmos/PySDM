# pylint:disable=missing-module-docstring
import os
import sys

from setuptools import setup

CI = "CI" in os.environ

dependencies = [
    "ThrustRTC>=0.3.20",
    "CURandRTC>=0.1.2",
    "numba>=0.51.2",
    # TODO #1344: (numpy 2.0.0 incompatibility in https://github.com/bjodah/chempy/issues/234)
    "numpy"
    + (
        {
            8: "==1.24.4",
            9: "==1.24.4",
            10: "==1.24.4",
            11: "==1.24.4",
            12: "==1.26.4",
            13: "==1.26.4",
        }[sys.version_info.minor]
        if CI
        else ""
    ),
    "Pint",
    "chempy",
    "scipy"
    + (
        {
            8: "==1.10.1",
            9: "==1.10.1",
            10: "==1.10.1",
            11: "==1.10.1",
            12: "==1.13.0",
            13: "==1.13.0",
        }[sys.version_info.minor]
        if CI
        else ""
    ),
    "pyevtk",
    "pyparsing" + ("==3.2.5" if CI else ""),
]

optional_dependencies = {
    "unit-tests": ["pytest", "pytest-timeout", "matplotlib!=3.9.1"],
    "nonunit-tests": ["pytest", "PySDM-examples", "PyPartMC"],
    "CI_version_pins": [
        "PyPartMC==1.7.2",
        "numba==0.60.0",
        "CURandRTC==0.1.7",
        "Pint==0.21.1",
        "chempy==0.8.3",
        "pyevtk==1.6.0",
    ],
}

setup(
    install_requires=dependencies,
    extras_require=optional_dependencies,
)

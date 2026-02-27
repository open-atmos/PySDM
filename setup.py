# pylint:disable=missing-module-docstring
import os
import sys

from setuptools import setup

CI = "CI" in os.environ

dependencies = [
    "ThrustRTC>=0.3.20",
    "CURandRTC>=0.1.2",
    "numba>=0.51.2",
    "numpy",
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
        "Pint==0.24.4",
        "chempy==0.8.3",
        "pyevtk==1.6.0",
    ],
}

setup(
    install_requires=dependencies,
    extras_require=optional_dependencies,
)

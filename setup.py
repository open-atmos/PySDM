# pylint:disable=missing-module-docstring
import os
import sys

from setuptools import setup

CI = "CI" in os.environ

dependencies = [
    "ThrustRTC==0.3.20",
    "CURandRTC" + ("==0.1.6" if CI else ">=0.1.2"),
    "numba"
    + (
        {
            8: "==0.58.1",
            9: "==0.60.0",
            10: "==0.60.0",
            11: "==0.60.0",
            12: "==0.60.0",
            13: "==0.60.0",
        }[sys.version_info.minor]
        if CI
        else ">=0.51.2"
    ),
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
    "Pint" + ("==0.21.1" if CI else ""),
    "chempy" + ("==0.8.3" if CI else ""),
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
    "pyevtk" + ("==1.6.0" if CI else ""),
]

optional_dependencies = {
    "tests": [
        "matplotlib",
        "Pillow"
        + (
            "<11.3.0" if CI else ""
        ),  # matplotlib triggers deprecation warnings in 11.3.0
        "pytest",
        "pytest-timeout",
        "PySDM-examples",
        "open-atmos-jupyter-utils>=v1.2.0",
    ]
    + ["PyPartMC" + ("==1.7.2" if CI else "")]
    + (
        [
            "pyrcel",
            "jupyter-core<5.0.0",
            "ipywidgets!=8.0.3",
        ]
    )
}

setup(
    install_requires=dependencies,
    extras_require=optional_dependencies,
)

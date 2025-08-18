# pylint:disable=missing-module-docstring
import os

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
    "chempy",
    "scipy",
    "pyevtk",
]

optional_dependencies = {
    "unit-tests": [
        "pytest",
        "matplotlib",
    ],
    "nonunit-tests": [
        "matplotlib",
        "Pillow",
        "pytest",
        "PySDM-examples",
        "open-atmos-jupyter-utils",
        "PyPartMC",
        "pyrcel",
        "jupyter-core",
        "ipywidgets",
    ],
    "CI_version_pins": [
        "PySDM-examples[CI_version_pins]",
        "Pillow<11.3.0",
        "PyPartMC==1.7.2",
        "open-atmos-jupyter-utils>=v1.2.0",
        "jupyter-core<5.0.0",
        "ipywidgets!=8.0.3",
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

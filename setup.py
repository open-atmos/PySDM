"""
the magick behind "pip install PySDM"
"""

import os
import platform
import sys

from setuptools import find_packages, setup


def get_long_description():
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
    return long_description


CI = "CI" in os.environ
_32bit = platform.architecture()[0] == "32bit"

setup(
    name="PySDM",
    description="Pythonic particle-based (super-droplet) warm-rain/aqueous-chemistry"
    " cloud microphysics package with box, parcel & 1D/2D prescribed-flow"
    " examples in Python, Julia and Matlab",
    use_scm_version={"local_scheme": lambda _: "", "version_scheme": "post-release"},
    install_requires=[
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
            }[sys.version_info.minor]
            if CI and not _32bit
            else ">=0.51.2"
        ),
        # TODO #1344: (numpy 2.0.0 incompatibility in https://github.com/bjodah/chempy/issues/234)
        "numpy" + ("==1.24.4" if CI else "<2.0.0"),
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
            }[sys.version_info.minor]
            if CI and not _32bit
            else ""
        ),
        "pyevtk" + ("==1.2.0" if CI else ""),
    ],
    extras_require={
        "tests": [
            "matplotlib",
            "pytest",
            "pytest-timeout",
            "PyPartMC==1.3.3",
        ]
        + (
            [
                "pywinpty" + ("==0.5.7" if CI else ""),
                "terminado" + ("==0.9.5" if CI else ""),
                "jupyter-client" + ("==7.4.9" if CI else ""),
                "jupyter-core" + ("==4.12.0" if CI else ""),
                "jupyter-server" + ("==1.24.0" if CI else ""),
                "notebook" + ("==6.5.6" if CI else ""),
            ]
            if _32bit
            else [
                "pyrcel",
                "jupyter-core<5.0.0",
                "ipywidgets!=8.0.3",
            ]
        )
    },
    author="https://github.com/open-atmos/PySDM/graphs/contributors",
    author_email="sylwester.arabas@agh.edu.pl",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/open-atmos/PySDM",
    license="GPL-3.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
    ],
    keywords="physics-simulation, monte-carlo-simulation, gpu-computing,"
    " atmospheric-modelling, particle-system, numba, thrust,"
    " nvrtc, pint, atmospheric-physics",
    packages=find_packages(include=["PySDM", "PySDM.*"]),
    project_urls={
        "Tracker": "https://github.com/open-atmos/PySDM/issues",
        "Documentation": "https://open-atmos.github.io/PySDM",
        "Source": "https://github.com/open-atmos/PySDM",
    },
)

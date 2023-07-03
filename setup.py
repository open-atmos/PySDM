"""
the magick behind "pip install PySDM"
"""
import os

from setuptools import find_packages, setup


def get_long_description():
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
    return long_description


CI = "CI" in os.environ

setup(
    name="PySDM",
    description="Pythonic particle-based (super-droplet) warm-rain/aqueous-chemistry"
    " cloud microphysics package with box, parcel & 1D/2D prescribed-flow"
    " examples in Python, Julia and Matlab",
    use_scm_version={"local_scheme": lambda _: "", "version_scheme": "post-release"},
    setup_requires=["setuptools_scm"],
    install_requires=[
        "ThrustRTC==0.3.20",
        "CURandRTC" + ("==0.1.6" if CI else ">=0.1.2"),
        "numba" + ("==0.56.4" if CI else ">=0.51.2"),
        "numpy" + ("==1.21.6" if CI else ""),
        "Pint" + ("==0.17" if CI else ""),
        "chempy" + ("==0.7.10" if CI else ""),
        "scipy" + ("==1.7.3" if CI else ""),
        "pyevtk" + ("==1.2.0" if CI else ""),
    ],
    extras_require={
        "tests": [
            "matplotlib" + ("==3.5.3" if CI else ""),
            "jupyter-core<5.0.0",
            "ipywidgets!=8.0.3",
            "ghapi",
            "pytest",
            "pytest-timeout",
            "PyPartMC==0.3.0",
        ]
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

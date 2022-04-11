"""
the magick behind "pip install PySDM"
"""
import os
from setuptools import setup, find_packages


def get_long_description():
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
    return long_description


setup(
    name='PySDM',
    description='Pythonic particle-based (super-droplet) warm-rain/aqueous-chemistry'
                ' cloud microphysics package with box, parcel & 1D/2D prescribed-flow'
                ' examples in Python, Julia and Matlab',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'ThrustRTC==0.3.19',
        'CURandRTC' + ('==0.1.6' if 'CI' in os.environ else '>=0.1.2'),
        'numba' + ('==0.55.0' if 'CI' in os.environ else '>=0.51.2'),
        'numpy' + ('==1.21' if 'CI' in os.environ else ''),
        'Pint' + ('==0.17' if 'CI' in os.environ else ''),
        'chempy' + ('==0.7.10' if 'CI' in os.environ else ''),
        'scipy' + ('==1.6.3' if 'CI' in os.environ else ''),
        'pyevtk' + ('==1.2.0' if 'CI' in os.environ else '')
    ],
    author='https://github.com/atmos-cloud-sim-uj/PySDM/graphs/contributors',
    author_email='sylwester.arabas@uj.edu.pl',
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/atmos-cloud-sim-uj/PySDM",
    license="GPL-3.0",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries'
    ],
    keywords='physics-simulation, monte-carlo-simulation, gpu-computing,'
             ' atmospheric-modelling, particle-system, numba, thrust,'
             ' nvrtc, pint, atmospheric-physics',
    packages=find_packages(include=['PySDM', 'PySDM.*'])
)

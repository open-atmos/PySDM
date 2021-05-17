from setuptools import setup, find_packages
import os


def get_long_description():
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
    return long_description


setup(
    name='PySDM',
    description='Pythonic particle-based (super-droplet) cloud microphysics modelling with Jupyter examples',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'ThrustRTC' + ('==0.3.10' if 'CI' in os.environ else ''),
        'CURandRTC' + ('==0.1.6' if 'CI' in os.environ else ''),
        'numba' + ('==0.53.1' if 'CI' in os.environ else ''),
        'numpy' + ('==1.20.2' if 'CI' in os.environ else ''),
        'Pint' + ('==0.17' if 'CI' in os.environ else ''),
        'chempy' + ('==0.7.10' if 'CI' in os.environ else ''),
        'scipy' + ('==1.6.3' if 'CI' in os.environ else '')
    ],
    author='https://github.com/atmos-cloud-sim-uj/PySDM/graphs/contributors',
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
    keywords='physics-simulation, monte-carlo-simulation, gpu-computing, atmospheric-modelling, particle-system, numba, thrust, nvrtc, pint, atmospheric-physics',
    packages=find_packages(include=['PySDM', 'PySDM.*'])
)

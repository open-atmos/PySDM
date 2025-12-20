import os
import re

from setuptools import find_packages, setup


def get_long_description():
    """returns contents of the pdoc landing site with pdoc links converted into URLs"""
    with open("docs/pysdm_examples_landing.md", "r", encoding="utf8") as file:
        pdoc_links = re.compile(
            r"(`)([\w\d_-]*).([\w\d_-]*)(`)", re.MULTILINE | re.UNICODE
        )
        return pdoc_links.sub(
            r'<a href="https://open-atmos.github.io/PySDM/\2/\3.html">\3</a>',
            file.read(),
        )


CI = "CI" in os.environ

setup(
    name="pysdm-examples",
    description="PySDM usage examples reproducing results from literature "
    "and depicting how to use PySDM from Python Jupyter notebooks",
    install_requires=[
        "PySDM",
        "PyMPDATA",
        "open-atmos-jupyter-utils",
        "pystrict",
        "matplotlib",
        "joblib",
        "ipywidgets",
        "seaborn",
        "numdifftools",
        "vtk",
        "pyrcel",
        "pyvinecopulib",
        "networkx",
    ],
    extras_require={
        "CI_version_pins": [
            "PySDM[CI_version_pins]",
            "PyMPDATA==1.7.0",
            "open-atmos-jupyter-utils==1.3.0",
            "pystrict==1.3",
            "matplotlib!=3.9.1",
            "joblib==1.5.3",
            "ipywidgets==8.1.7",
            "seaborn==0.13.2",
            "numdifftools==0.9.42",
            "vtk==9.5.2",
            "pyrcel==1.3.4",
            "pyvinecopulib==0.7.3",
        ]
    },
    author="https://github.com/open-atmos/PySDM/graphs/contributors",
    author_email="sylwester.arabas@agh.edu.pl",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/open-atmos/PySDM",
    license="GPL-3.0",
    packages=find_packages(include=["PySDM_examples", "PySDM_examples.*"]),
    project_urls={
        "Tracker": "https://github.com/open-atmos/PySDM/issues",
        "Documentation": "https://open-atmos.github.io/PySDM/PySDM_examples",
        "Source": "https://github.com/open-atmos/PySDM/tree/main/examples",
    },
)

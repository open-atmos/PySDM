import os
import platform

from setuptools import find_packages, setup


def get_long_description():
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
    return long_description


CI = "CI" in os.environ

setup(
    name="PySDM-examples",
    description="PySDM usage examples reproducing results from literature "
    "and depicting how to use PySDM from Python Jupyter notebooks",
    use_scm_version={
        "local_scheme": lambda _: "",
        "version_scheme": "post-release",
        "root": "..",
    },
    setup_requires=["setuptools_scm"],
    install_requires=[
        "PySDM",
        "PyMPDATA",
        "open-atmos-jupyter-utils",
        "pystrict",
        "matplotlib",
        "joblib",
        "ipywidgets",
        "seaborn",
        "ghapi",
    ]
    + (["pyvinecopulib", "vtk"] if platform.architecture()[0] != "32bit" else []),
    extras_require={
        "tests": [
            "pytest",
            "nbconvert",
            "jupyter-core" + "<5.0.0" if CI else "",
            # https://github.com/jupyter/nbformat/issues/232
            "jsonschema" + "==3.2.0" if CI else "",
            # https://github.com/jupyter/nbconvert/issues/1568
            "Jinja2" + "<3.0.0" if CI else "",
            # https://github.com/aws/aws-sam-cli/issues/3661
            "MarkupSafe" + "<2.1.0" if CI else "",
            # github.com/python-pillow/Pillow/blob/main/docs/releasenotes/9.1.0.rst#deprecations
            "Pillow" + "<9.1.0" if CI else "",
            "ipywidgets" + "<8.0.3" if CI else "",
            # https://github.com/dask/distributed/issues/7688
            "ipykernel" + "<6.22.0" if CI else "",
        ]
    },
    author="https://github.com/open-atmos/PySDM/graphs/contributors",
    author_email="slayoo@staszic.waw.pl",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/open-atmos/PySDM-examples",
    license="GPL-3.0",
    packages=find_packages(include=["PySDM_examples", "PySDM_examples.*"]),
    project_urls={
        "Tracker": "https://github.com/open-atmos/PySDM/issues",
        "Documentation": "https://open-atmos.github.io/PySDM-examples",
        "Source": "https://github.com/open-atmos/PySDM-examples",
    },
)

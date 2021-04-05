import exdown
import pathlib
import pytest
import platform

# note: skipping Windows to avoid non-UTF encoding trouble
test_readme = pytest.mark.skipif('platform.system() == "Windows"')(exdown.pytests(
    pathlib.Path(__file__).parent.parent.parent.joinpath("README.md").absolute(),
    syntax_filter="Python"
))

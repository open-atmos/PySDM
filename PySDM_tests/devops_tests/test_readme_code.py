import exdown
import pathlib
import pytest
import sys


test_readme = pytest.mark.skipif("sys.getfilesystemencoding() != 'utf-8'")(exdown.pytests(
    pathlib.Path(__file__).parent.parent.parent.joinpath("README.md").absolute(),
    syntax_filter="Python"
))

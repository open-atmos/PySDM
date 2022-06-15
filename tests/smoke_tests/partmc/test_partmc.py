# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import sys

import pytest

if sys.platform == "linux":
    import PyPartMC


@pytest.mark.skipif(sys.platform != "linux", reason="binary package availability")
def test_partmc():
    print(PyPartMC.__version__)

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import platform

import pytest

if platform.architecture()[0] == "64bit":
    import PyPartMC


@pytest.mark.skipif(
    platform.architecture()[0] != "64bit", reason="binary package availability"
)
def test_partmc():
    print(PyPartMC.__version__)

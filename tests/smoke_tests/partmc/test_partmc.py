# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import platform
import pytest

if platform.machine().endswith('64'):
    import PyPartMC


@pytest.mark.skipif(not platform.machine().endswith('64'), reason="binary package availability")
def test_partmc():
    print(PyPartMC.__version__)

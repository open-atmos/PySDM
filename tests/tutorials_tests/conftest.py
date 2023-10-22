import pathlib

import pytest

from ..examples_tests.conftest import findfiles

PYSDM_TUTORIALS_ABS_PATH = (
    pathlib.Path(__file__).parent.parent.parent.absolute().joinpath("tutorials")
)


@pytest.fixture(
    params=(path for path in findfiles(PYSDM_TUTORIALS_ABS_PATH, r".*\.(ipynb)$")),
)
def notebook_filename(request):
    return request.param


@pytest.fixture(
    params=(path for path in findfiles(PYSDM_TUTORIALS_ABS_PATH, r".*\.(py)$")),
)
def tutorial_filename(request):
    return request.param

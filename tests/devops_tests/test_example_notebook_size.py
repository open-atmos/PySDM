""" checks if all example Jupyter notebooks have file size less than a certain limit """
import os
import pathlib

import pytest

from PySDM.physics import si
from tests.devops_tests.test_todos_annotated import findfiles


@pytest.fixture(
    params=findfiles(
        (pathlib.Path(__file__).parent.parent.parent / "PySDM-examples").absolute(),
        r".*\.(ipynb)$",
    ),
    name="notebook",
)
def _notebook(request):
    return request.param


def test_example_notebook_size(notebook):
    """returns True if a given notebook matches file size criterion"""
    assert os.stat(notebook).st_size * si.byte < 4.5 * si.megabyte  # TODO #1074

# pylint: disable=missing-module-docstring
from ..devops_tests.test_notebooks import test_run_notebooks as _impl


def test_run_notebooks(notebook_filename, tmp_path):
    _impl(notebook_filename, tmp_path)

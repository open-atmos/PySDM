"""
test notebook_vars import is from open-atmos-jupyter-utils
"""

import pytest

from pathlib import Path
import pytest

from open_atmos_jupyter_utils import notebook_vars
import examples


@pytest.fixture(scope="session", name="notebook_variables")
def notebook_variables_fixture():
    """returns variables from the notebook"""
    print(examples.__file__)
    return notebook_vars(
        file=Path(examples.__file__).parent / "show_anim_example.ipynb",
        plot=False,
    )


def test_notebook_vars(notebook_variables):
    assert notebook_variables["frame_range"][-1] == 49

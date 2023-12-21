# pylint: disable=missing-module-docstring
import pathlib


def test_run_tutorials(tutorial_filename):
    if pathlib.Path(tutorial_filename).name == "__init__.py":
        return

    with open(tutorial_filename, encoding="utf8") as f:
        exec(f.read(), {"__name__": "__main__"})  # pylint: disable=exec-used

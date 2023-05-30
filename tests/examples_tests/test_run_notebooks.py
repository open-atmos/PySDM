# pylint: disable=wrong-import-position
# https://bugs.python.org/issue37373
import sys

if sys.platform == "win32" and sys.version_info[:2] >= (3, 7):
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def test_run_notebooks(notebook_filename, tmp_path):
    with open(notebook_filename, encoding="utf8") as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=15 * 60, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": tmp_path}})

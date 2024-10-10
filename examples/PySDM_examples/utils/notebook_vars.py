""" helper routines for use in smoke tests """

from pathlib import Path

import nbformat


def notebook_vars(file: Path, plot: bool):
    """Executes the code from all cells of the Jupyter notebook `file` and
    returns a dictionary with the notebook variables. If the `plot` argument
    is set to `True`, any code line within the notebook starting with `show_plot(`
    (see [open_atmos_jupyter_utils docs](https://pypi.org/p/open_atmos_jupyter_utils))
    is replaced with `pyplot.show() #`, otherwise it is replaced with `pyplot.gca().clear() #`
    to match the smoke-test conventions."""
    notebook = nbformat.read(file, nbformat.NO_CONVERT)
    context = {}
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            lines = cell.source.splitlines()
            for i, line in enumerate(lines):
                if line.strip().startswith("!"):
                    lines[i] = line.replace("!", "pass #")
                if line.strip().startswith("show_plot("):
                    lines[i] = line.replace(
                        "show_plot(",
                        "from matplotlib import pyplot; "
                        + ("pyplot.show() #" if plot else "pyplot.gca().clear() #"),
                    )

            exec("\n".join(lines), context)  # pylint: disable=exec-used
    return context

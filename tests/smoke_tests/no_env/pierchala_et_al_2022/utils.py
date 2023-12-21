""" common notebook execution logic """
import nbformat

from PySDM.physics.constants_defaults import (  # pylint:disable=unused-import
    PER_MEG,
    PER_MILLE,
)


def notebook_vars(file, plot):
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
                        "pyplot.show() #" if plot else "pyplot.gca().clear() #",
                    )

            exec("\n".join(lines), context)  # pylint: disable=exec-used
    return context

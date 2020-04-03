import tempfile, os
from IPython.display import FileLink, display
from ipywidgets import HTML, VBox, Output
from matplotlib import pyplot


def show_plot(filename=None):
    pyplot.legend(loc = 'best')
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
    if filename is None:
        _, tempfile_path = tempfile.mkstemp(
            dir=outdir,
            suffix='.pdf'
        )
    else:
        tempfile_path = os.path.join(outdir, filename)
    pyplot.savefig(tempfile_path, type='pdf')

    output = Output()
    with output:
        pyplot.show()
    link = HTML()
    filename = str(os.path.join('../utils/output', os.path.basename(tempfile_path)))
    link.value = FileLink(filename)._format_path()
    display(VBox([output, link]))
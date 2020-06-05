import tempfile, os
from IPython.display import FileLink, display
from ipywidgets import HTML, VBox, Output
from matplotlib import pyplot


def make_link(fig, filename=None):
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'output')
    if filename is None:
        _, tempfile_path = tempfile.mkstemp(
            dir=outdir,
            suffix='.pdf'
        )
    else:
        tempfile_path = os.path.join(outdir, filename)
    fig.savefig(tempfile_path, type='pdf')
    link = HTML()
    filename = str(os.path.join('../utils/output', os.path.basename(tempfile_path)))
    link.value = FileLink(filename)._format_path()
    return link


def show_plot(filename=None):
    pyplot.legend(loc = 'best')

    output = Output()
    with output:
        pyplot.show()
    link = make_link(pyplot, filename)
    display(VBox([output, link]))
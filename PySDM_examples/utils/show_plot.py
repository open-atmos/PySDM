from .temporary_file import TemporaryFile
from matplotlib import pyplot
from IPython.display import display


def show_plot(filename=None):
    link = save_and_make_link(pyplot, filename)
    pyplot.show()
    display(link)


def save_and_make_link(fig, filename=None):
    temporary_file = TemporaryFile(suffix='.pdf', filename=filename)
    fig.savefig(temporary_file.absolute_path)
    return temporary_file.make_link_widget()

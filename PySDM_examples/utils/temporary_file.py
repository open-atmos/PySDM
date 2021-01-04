import os, sys
import tempfile
from .widgets import FileLink, HTML, Button
if 'google.colab' in sys.modules:
    from google import colab

ABSOLUTE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'temporary_files')
RELATIVE_PATH = '../utils/temporary_files'


class TemporaryFile:
    def __init__(self, suffix='.pdf', filename=None):
        if filename is None:
            _, self.absolute_path = tempfile.mkstemp(
                dir=ABSOLUTE_PATH,
                suffix=suffix
            )
        else:
            self.absolute_path = os.path.join(ABSOLUTE_PATH, filename)
        self.basename = os.path.basename(self.absolute_path)

    def make_link_widget(self):
        if 'google.colab' in sys.modules:
            link = Button(description=self.basename)
            link.on_click(lambda _:colab.files.download(self.absolute_path))
        else:
            link = HTML()
            filename = str(os.path.join(RELATIVE_PATH, self.basename))
            link.value = FileLink(filename)._format_path()
        return link
"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

from ipywidgets import FloatProgress, Button, HBox, HTML
from IPython.display import FileLink
from time import sleep
from threading import Thread


class DemoController:
    progress = FloatProgress(value=0.0, min=0.0, max=1.0, description="%")
    button = Button()
    link = HTML()
    panic = False
    thread = None

    def __init__(self, simulator, viewer, exporter):
        self.simulator = simulator
        self.exporter = exporter
        self.ncdf_filename = exporter.filename
        self.viewer = viewer
        self._setup_play()

    def __enter__(self):
        self.panic = False
        self.set_percent(0)

    def __exit__(self, *_):
        if self.panic:
            self._setup_play()
        else:
            self._setup_ncdf()
        self.panic = False

    def reinit(self, _=None):
        self.panic = True
        self.viewer.reinit()
        self._setup_play()
        self.progress.value = 0
        self.link.value = ''

    def box(self):
        return HBox([self.progress, self.button, self.link])

    def set_percent(self, value):
        self.progress.value = value

    def _setup_play(self):
        self.button.on_click(self._handle_stop, remove=True)
        self.button.on_click(self._handle_ncdf, remove=True)
        self.button.on_click(self._handle_play)
        self.button.icon = 'play'

    def _setup_stop(self):
        self.button.on_click(self._handle_play, remove=True)
        self.button.on_click(self._handle_ncdf, remove=True)
        self.button.on_click(self._handle_stop)
        self.button.icon = 'stop'

    def _setup_ncdf(self):
        self.button.icon = 'download'
        self.button.on_click(self._handle_stop, remove=True)
        self.button.on_click(self._handle_ncdf)

    def _handle_stop(self, _):
        self.panic = True
        while self.panic: sleep(0.1)
        self._setup_play()

    def _handle_play(self, _):
        self._setup_stop()
        self.link.value = ''
        self.thread = Thread(target=self.simulator.run, args=(self,))
        self.thread.start()

    def _handle_ncdf(self, _):
        self.thread = Thread(target=self.exporter.run, args=(self,))
        self.thread.start()
        self.link.value = FileLink(self.ncdf_filename)._format_path()
        self._setup_stop()

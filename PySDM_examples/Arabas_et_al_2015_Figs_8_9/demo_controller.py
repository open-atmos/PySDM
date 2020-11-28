"""
Created at 02.10.2019
"""

from ..utils.widgets import FloatProgress, Button, HBox
from time import sleep
from threading import Thread


class DemoController:
    def __init__(self, simulator, viewer, exporter, ncdf_file):
        self.progress = FloatProgress(value=0.0, min=0.0, max=1.0)
        self.button = Button()
        self.link = HBox()
        self.panic = False
        self.thread = None
        self.simulator = simulator
        self.exporter = exporter
        self.ncdf_file = ncdf_file
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
        self.progress.description = ' '

    def reinit(self, _=None):
        self.panic = True
        self._setup_play()
        self.progress.value = 0
        self.progress.description = ' '
        self.viewer.clear()
        self.link.children = ()

    def box(self):
        return HBox([self.progress, self.button, self.link])

    def set_percent(self, value):
        self.progress.description = 'running'
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

        self.link.children = ()
        self.progress.description = 'initialisation'

        self.thread = Thread(target=self.simulator.run, args=(self,))

        self.simulator.reinit()
        self.thread.start()
        self.viewer.reinit(self.simulator.products)

    def _handle_ncdf(self, _):
        self.thread = Thread(target=self.exporter.run, args=(self,))
        self.thread.start()
        self.link.children = (self.ncdf_file.make_link_widget(),)
        self._setup_stop()

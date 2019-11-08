"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

from ipywidgets import FloatProgress, Button, HBox
from time import sleep
from threading import Thread


class DemoController:
    progress = FloatProgress(value=0.0, min=0.0, max=1.0, description="%")
    button = Button()
    panic = False
    thread = None

    def __init__(self, simulation, viewer):
        self.target = simulation.run
        self.viewer = viewer
        self._setup_play()

    def __enter__(self):
        self.panic = False

    def __exit__(self, *_):
        self.panic = False
        self._setup_play()

    def reinit(self, _=None):
        self.panic = True
        self.viewer.reinit()
        self._setup_play()
        self.progress.value = 0

    def box(self):
        return HBox([self.progress, self.button])

    def set_percent(self, value):
        self.progress.value = value

    def _setup_play(self):
        self.button.on_click(self._handle_stop, remove=True)
        self.button.on_click(self._handle_play)
        self.button.icon = 'play'

    def _setup_stop(self):
        self.button.on_click(self._handle_play, remove=True)
        self.button.on_click(self._handle_stop)
        self.button.icon = 'stop'

    def _handle_stop(self, _):
        self.panic = True
        while self.panic: sleep(0.1)
        self._setup_play()

    def _handle_play(self, _):
        self.thread = Thread(target=self.target, args=(self,))
        self.thread.start()
        self._setup_stop()
        # self.viewer.reinit()

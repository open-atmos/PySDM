from ipywidgets import VBox, Box, Play, Output, IntSlider, jslink
from matplotlib.pyplot import imshow, show, draw, colorbar
from IPython.display import clear_output, display
import numpy as np


class DemoViewer:
    play = Play()
    slider = IntSlider()
    plot = Output()

    def __init__(self, storage, setup):
        self.storage = storage
        self.setup = setup
        self.nans = None
        self.im = None
        self.reinit()

    def reinit(self, _=None):
        n_steps = len(self.setup.steps)
        self.slider.max = n_steps - 1
        self.play.max = n_steps - 1
        self.play.value = 0
        self.slider.value = 0
        self.nans = np.full((self.setup.grid[0], self.setup.grid[1]), np.nan)
        with self.plot:
            clear_output()
            self.im = imshow(self.nans, cmap='GnBu')
            self.im.set_clim(vmin=0, vmax=500000000)
            colorbar()
            show()

    def replot(self, bunch):
        step = bunch.new
        with self.plot:
            try:
                data = self.storage.load(self.setup.steps[step])
            except self.storage.Exception:
                data = self.nans
            self.im.set_data(data)
            clear_output(wait=True)
            display(self.im.figure)

    def box(self):
        jslink((self.play, 'value'), (self.slider, 'value'))
        self.play.observe(self.replot, 'value')
        return VBox([Box([self.play, self.slider]), self.plot])

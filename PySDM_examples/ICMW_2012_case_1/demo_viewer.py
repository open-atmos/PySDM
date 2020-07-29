"""
Created at 02.10.2019
"""

from ipywidgets import VBox, Box, Play, Output, IntSlider, IntRangeSlider, jslink, \
    HBox, Dropdown, Button, Layout
from IPython.display import clear_output, display
import numpy as np
from PySDM.physics import formulae as phys
from PySDM.physics import constants as const
from ..utils.show_plot import make_link

from .demo_plots import _ImagePlot, _SpectrumPlot


class DemoViewer:

    def __init__(self, storage, setup):
        self.storage = storage
        self.setup = setup

        self.play = Play(interval=10000)
        self.step_slider = IntSlider(continuous_update=False)
        self.product_select = Dropdown()
        self.plots_box = Box()

        self.slider = {}
        self.lines = {'x': [{}, {}], 'y': [{}, {}]}
        for xy in ('x', 'y'):
            self.slider[xy] = IntRangeSlider(min=0, max=1, description=f'spectrum_{xy}',
                                             continuous_update=False,
                                             orientation='horizontal' if xy == 'x' else 'vertical')

        self.reinit({})

    def clear(self):
        self.plots_box.children = ()

    def reinit(self, products):
        self.products = products
        self.product_select.options = [
            (f"{val.description} [{val.unit}]", key)
            for key, val in products.items()
            if len(val.shape) == 2
        ]

        r_bins = phys.radius(volume=self.setup.v_bins)
        const.convert_to(r_bins, const.si.micrometres)
        self.spectrumOutput = Output()
        with self.spectrumOutput:
            clear_output()
            self.spectrumPlot = _SpectrumPlot(r_bins)

        self.plots = {}
        self.outputs = {}
        for key, product in products.items():
            if len(product.shape) == 2:
                self.outputs[key] = Output()
                with self.outputs[key]:
                    clear_output()
                    self.plots[key] = _ImagePlot(self.setup.grid, self.setup.size, product)

        self.plot_box = Box()
        if len(products.keys()) > 0:
            layout_flex_end = Layout(display='flex', justify_content='flex-end')
            save_map = Button(icon='save')
            save_map.on_click(self.handle_save_map)
            save_spe = Button(icon='save')
            save_spe.on_click(self.handle_save_spe)
            self.plots_box.children = (
                HBox(
                    children=(
                        VBox(
                            children=(
                                Box(
                                    layout=layout_flex_end,
                                    children=(save_map,)
                                ),
                                HBox((self.slider['y'], self.plot_box)),
                                HBox((self.slider['x'],), layout=layout_flex_end)
                            )
                        ),
                        VBox(
                            layout=Layout(),
                            children=(save_spe, self.spectrumOutput)
                        )
                    )
                ),
            )

        for widget in (self.step_slider, self.play):
            widget.value = 0
            widget.max = len(self.setup.steps) - 1

        for j, xy in enumerate(('x', 'y')):
            slider = self.slider[xy]
            mx = self.setup.grid[j]
            slider.max = mx
            slider.value = (0, mx)

        self.replot()

    def handle_save_map(self, _):
        display(make_link(self.plots[self.product_select.value].fig))

    def handle_save_spe(self, _):
        display(make_link(self.spectrumPlot.fig))

    def replot(self, *args, **kwargs):
        self.replot_image()
        self.replot_spectra()

    def replot_spectra(self, *args, **kwargs):
        step = self.step_slider.value

        xrange = slice(*self.slider['x'].value)
        yrange = slice(*self.slider['y'].value)

        for key in ('Particles Wet Size Spectrum', 'Particles Dry Size Spectrum'):
            try:
                data = self.storage.load(self.setup.steps[step], key)
                data = data[xrange, yrange, :]
                data = np.mean(np.mean(data, axis=0), axis=0)
                data = np.concatenate(((0,), data))
                if key == 'Particles Wet Size Spectrum':
                    self.spectrumPlot.update_wet(data, step)
                if key == 'Particles Dry Size Spectrum':
                    self.spectrumPlot.update_dry(data)
            except self.storage.Exception:
                pass

        with self.spectrumOutput:
            clear_output(wait=True)
            display(self.spectrumPlot.fig)

        selected = self.product_select.value
        if selected is None or selected not in self.plots:
            return

        self.plots[selected].update(None, self.slider['x'].value, self.slider['y'].value, step)

        for key in self.outputs.keys():
            with self.outputs[key]:
                clear_output(wait=True)
                display(self.plots[key].fig)

    def replot_image(self, *args, **kwargs):
        selected = self.product_select.value

        if selected is None or selected not in self.plots:
            return

        if selected in self.outputs:
            self.plot_box.children = [self.outputs[selected]]

        step = self.step_slider.value
        try:
            data = self.storage.load(self.setup.steps[step], selected)
        except self.storage.Exception:
            data = None

        self.plots[selected].update(data, self.slider['x'].value, self.slider['y'].value, step)

        for key in self.outputs.keys():
            with self.outputs[key]:
                clear_output(wait=True)
                display(self.plots[key].fig)

    def box(self):
        jslink((self.play, 'value'), (self.step_slider, 'value'))
        self.step_slider.observe(self.replot, 'value')
        self.product_select.observe(self.replot_image, 'value')
        for xy in ('x', 'y'):
            self.slider[xy].observe(self.replot_spectra, 'value')
        return VBox([
            Box([self.play, self.step_slider, self.product_select]),
            self.plots_box
        ])

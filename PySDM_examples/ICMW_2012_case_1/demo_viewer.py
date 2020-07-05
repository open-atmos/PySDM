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

        self.play = Play(interval=1000)
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
        self.plots = {}
        self.outputs = {}
        for key in products.keys():
            self.outputs[key] = Output()

            with self.outputs[key]:
                clear_output()
                product = self.products[key]
                if len(product.shape) == 2:
                    self.plots[key] = _ImagePlot(self.setup.grid, self.setup.size, product)
                elif len(product.shape) == 3:
                    r_bins = phys.radius(volume=self.setup.v_bins)
                    const.convert_to(r_bins, const.si.micrometres)
                    self.plots[key] = _SpectrumPlot(r_bins)
                else:
                    raise NotImplementedError()

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
                            children=(save_spe, self.outputs['Particles Size Spectrum'])
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
        display(make_link(self.plots['Particles Size Spectrum'].fig))

    def replot(self, _=None):
        selected = self.product_select.value

        if selected in self.outputs:
            self.plot_box.children = [self.outputs[selected]]

        step = self.step_slider.value
        for key in self.outputs.keys():
            if len(self.products[key].shape) == 2:
                if key != selected:
                    continue

                try:
                    data = self.storage.load(self.setup.steps[step], key)
                except self.storage.Exception:
                    data = None

                self.plots[key].update(data, self.slider['x'].value, self.slider['y'].value)
            elif len(self.products[key].shape) == 3:
                xrange = slice(*self.slider['x'].value)
                yrange = slice(*self.slider['y'].value)
                try:
                    data = self.storage.load(self.setup.steps[step], key)
                    data = data[xrange, yrange, :]
                    data = np.mean(np.mean(data, axis=0), axis=0)
                    data = np.concatenate(((0,), data))
                    self.plots[key].update(data)
                except self.storage.Exception:
                    pass
            else:
                raise NotImplementedError()

        for key in self.outputs.keys():
            with self.outputs[key]:
                clear_output(wait=True)
                display(self.plots[key].fig)

    def box(self):
        jslink((self.play, 'value'), (self.step_slider, 'value'))
        self.play.observe(self.replot, 'value')
        self.product_select.observe(self.replot, 'value')
        for xy in ('x', 'y'):
            self.slider[xy].observe(self.replot, 'value')
        return VBox([
            Box([self.play, self.step_slider, self.product_select]),
            self.plots_box
        ])

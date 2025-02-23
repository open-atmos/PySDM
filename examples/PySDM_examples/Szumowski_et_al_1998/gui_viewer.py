import warnings

import numpy as np
from matplotlib import pyplot, rcParams
from open_atmos_jupyter_utils import save_and_make_link
from PySDM_examples.Szumowski_et_al_1998.plots import (
    _ImagePlot,
    _SpectrumPlot,
    _TemperaturePlot,
    _TerminalVelocityPlot,
    _TimeseriesPlot,
)
from PySDM_examples.utils.widgets import (
    Box,
    Button,
    Dropdown,
    HBox,
    IntRangeSlider,
    IntSlider,
    Layout,
    Output,
    Play,
    VBox,
    clear_output,
    display,
    jslink,
)

from PySDM.physics import constants as const


class GUIViewer:
    def __init__(self, storage, settings):
        self.storage = storage
        self.settings = settings

        self.play = Play(interval=1000)
        self.step_slider = IntSlider(continuous_update=False, description="t/dt_out:")
        self.product_select = Dropdown()
        self.spectrum_select = Dropdown()
        self.plots_box = Box()

        self.timeseriesOutput = None
        self.timeseriesPlot = None
        self.plot_box = None
        self.spectrum_box = None
        self.outputs = None
        self.plots = None
        self.spectrumOutputs = None
        self.spectrumPlots = None
        self.products = None

        self.slider = {}
        self.lines = {"X": [{}, {}], "Z": [{}, {}]}
        for xz in ("X", "Z"):
            self.slider[xz] = IntRangeSlider(
                min=0,
                max=1,
                description=f"{xz}",
                continuous_update=False,
                orientation="horizontal" if xz == "X" else "vertical",
            )

        self.reinit({})

    def clear(self):
        self.plots_box.children = ()

    def reinit(self, products):
        self.products = products
        self.product_select.options = tuple(
            (f"{val.name} [{val.unit}]", key)
            for key, val in sorted(self.products.items(), key=lambda item: item[1].name)
            if len(val.shape) == 2
        )
        opts = [
            ("particle size spectra", "size"),
            ("terminal velocity", "terminal velocity"),
        ]
        if "freezable specific concentration" in products:
            opts.append(("freezing temperature spectra", "temperature"))
        if "immersed surface area" in products:
            pass  # TODO #599
        self.spectrum_select.options = tuple(opts)

        r_bins = self.settings.r_bins_edges.copy()
        const.convert_to(r_bins, const.si.micrometres)
        self.spectrumOutputs = {}
        self.spectrumPlots = {}
        for key in ("size", "terminal velocity", "temperature"):
            self.spectrumOutputs[key] = Output()
            with self.spectrumOutputs[key]:
                self.spectrumPlots[key] = (
                    _SpectrumPlot(r_bins, self.settings.spectrum_per_mass_of_dry_air)
                    if key == "size"
                    else (
                        _TemperaturePlot(
                            self.settings.T_bins_edges, self.settings.formulae
                        )
                        if key == "temperature"
                        else _TerminalVelocityPlot(
                            self.settings.terminal_velocity_radius_bin_edges,
                            self.settings.formulae,
                        )
                    )
                )
                clear_output()

        self.timeseriesOutput = Output()
        with self.timeseriesOutput:
            default_figsize = rcParams["figure.figsize"]
            fig_kw = {"figsize": (2.5 * default_figsize[0], default_figsize[1] / 2)}
            fig, ax = pyplot.subplots(1, 1, **fig_kw)
            self.timeseriesPlot = _TimeseriesPlot(
                fig, ax, self.settings.output_steps * self.settings.dt
            )
            clear_output()

        self.plots = {}
        self.outputs = {}
        for key, product in products.items():
            if len(product.shape) == 2:
                self.outputs[key] = Output()
                with self.outputs[key]:
                    fig, ax = pyplot.subplots(1, 1)
                    self.plots[key] = _ImagePlot(
                        fig,
                        ax,
                        self.settings.grid,
                        self.settings.size,
                        product,
                        show=True,
                        lines=True,
                    )
                    clear_output()

        self.plot_box = Box()
        self.spectrum_box = Box()
        if len(products.keys()) > 0:
            layout_flex_end = Layout(display="flex", justify_content="flex-end")
            save_map = Button(icon="save")
            save_map.on_click(self.handle_save_map)
            save_spe = Button(icon="save")
            save_spe.on_click(self.handle_save_spe)
            self.plots_box.children = (
                VBox(
                    children=(
                        HBox(
                            children=(
                                VBox(
                                    children=(
                                        Box(
                                            layout=layout_flex_end,
                                            children=(save_map, self.product_select),
                                        ),
                                        HBox((self.slider["Z"], self.plot_box)),
                                        HBox(
                                            (self.slider["X"],), layout=layout_flex_end
                                        ),
                                    )
                                ),
                                VBox(
                                    layout=Layout(),
                                    children=(
                                        Box(
                                            children=(save_spe, self.spectrum_select),
                                            layout=layout_flex_end,
                                        ),
                                        self.spectrum_box,
                                    ),
                                ),
                            )
                        ),
                        HBox((self.timeseriesOutput,)),
                    )
                ),
            )

        for widget in (self.step_slider, self.play):
            widget.value = 0
            widget.max = len(self.settings.output_steps) - 1

        for j, xz in enumerate(("X", "Z")):
            slider = self.slider[xz]
            mx = self.settings.grid[j]
            slider.max = mx
            slider.value = (0, mx)

        self.replot()

    def handle_save_map(self, _):
        display(save_and_make_link(self.plots[self.product_select.value].fig))

    def handle_save_spe(self, _):
        display(save_and_make_link(self.spectrumPlots[self.spectrum_select.value].fig))

    def replot(self, *_):
        selectedImage = self.product_select.value
        if not (selectedImage is None or selectedImage not in self.plots):
            self.replot_image()
            self.outputs[selectedImage].clear_output(wait=True)
            with self.outputs[selectedImage]:
                display(self.plots[selectedImage].fig)

        selectedSpectrum = self.spectrum_select.value
        if not (selectedSpectrum is None or selectedSpectrum not in self.spectrumPlots):
            self.replot_spectra()
            self.spectrumOutputs[selectedSpectrum].clear_output(wait=True)
            with self.spectrumOutputs[selectedSpectrum]:
                display(self.spectrumPlots[selectedSpectrum].fig)

        self.update_timeseries()
        self.timeseriesOutput.clear_output(wait=True)
        with self.timeseriesOutput:
            display(self.timeseriesPlot.fig)

    def update_spectra(self):
        selected = self.spectrum_select.value
        self.spectrum_box.children = [self.spectrumOutputs[selected]]
        plot = self.spectrumPlots[selected]

        step = self.step_slider.value

        xrange = slice(*self.slider["X"].value)
        yrange = slice(*self.slider["Z"].value)

        if selected == "size":
            for key in ("Particles Wet Size Spectrum", "Particles Dry Size Spectrum"):
                if xrange.start == xrange.stop or yrange.start == yrange.stop:
                    continue
                try:
                    data = self.storage.load(key, self.settings.output_steps[step])
                    data = data[xrange, yrange, :]
                    data = np.mean(np.mean(data, axis=0), axis=0)
                    data = np.concatenate(((0,), data))
                    if key == "Particles Wet Size Spectrum":
                        plot.update_wet(data, step)
                    if key == "Particles Dry Size Spectrum":
                        plot.update_dry(data)
                except self.storage.Exception:
                    pass
        elif selected == "terminal velocity":
            try:
                data = self.storage.load(
                    "radius binned number averaged terminal velocity",
                    self.settings.output_steps[step],
                )

                data = data[xrange, yrange, :]
                data = np.where(data != 0, data, np.nan)
                try:
                    with warnings.catch_warnings(record=True) as _:
                        warnings.simplefilter("ignore")
                        data_min = np.nanmin(np.nanmin(data, axis=0), axis=0)
                        data_max = np.nanmax(np.nanmax(data, axis=0), axis=0)
                except RuntimeWarning:
                    pass
                plot.update(data_min, data_max, step)
            except self.storage.Exception:
                pass
        elif selected == "temperature":
            try:
                dT = abs(self.settings.T_bins_edges[1] - self.settings.T_bins_edges[0])
                np.testing.assert_allclose(np.diff(self.settings.T_bins_edges), dT)

                conc = self.storage.load(
                    "particle specific concentration", self.settings.output_steps[step]
                )
                # TODO #705: assert unit == mg^-1
                conc = conc[xrange, yrange]

                data = self.storage.load(
                    "freezable specific concentration", self.settings.output_steps[step]
                )
                data = data[xrange, yrange, :]

                data = np.sum(np.sum(data, axis=0), axis=0) / np.sum(
                    np.sum(conc, axis=0), axis=0
                )
                data = np.concatenate(((0,), dT * np.cumsum(data[::-1])))[::-1]

                plot.update(data, step)
            except self.storage.Exception:
                pass
        else:
            raise NotImplementedError()

    def replot_spectra(self, *_):
        self.update_spectra()

        selected = self.product_select.value
        if selected is None or selected not in self.plots:
            return
        self.plots[selected].update_lines(
            self.slider["X"].value, self.slider["Z"].value
        )

        self.outputs[selected].clear_output(wait=True)

        key = self.spectrum_select.value
        self.spectrumOutputs[key].clear_output(wait=True)
        with self.outputs[selected]:
            display(self.plots[selected].fig)
        with self.spectrumOutputs[key]:
            display(self.spectrumPlots[key].fig)

    def update_image(self):
        selected = self.product_select.value

        if selected in self.outputs:
            self.plot_box.children = [self.outputs[selected]]

        step = self.step_slider.value
        try:
            data = self.storage.load(selected, self.settings.output_steps[step])
        except self.storage.Exception:
            data = None

        self.plots[selected].update(
            data, step, self.storage.data_range(selected) if data is not None else None
        )

    def replot_image(self, *_):
        selected = self.product_select.value
        if selected is None or selected not in self.plots:
            return

        self.update_image()
        self.outputs[selected].clear_output(wait=True)
        with self.outputs[selected]:
            display(self.plots[selected].fig)

    def update_timeseries(self):
        try:
            data = self.storage.load("surf_precip")
        except self.storage.Exception:
            data = None
        self.timeseriesPlot.update(
            data, self.storage.data_range("surf_precip") if data is not None else None
        )

    def replot_timeseries(self):
        self.update_timeseries()
        self.timeseriesOutput.clear_output(wait=True)
        with self.timeseriesOutput:
            display(self.timeseriesPlot.fig)

    def box(self):
        jslink((self.play, "value"), (self.step_slider, "value"))
        self.step_slider.observe(self.replot, "value")
        self.product_select.observe(self.replot_image, "value")
        self.spectrum_select.observe(self.replot_spectra, "value")
        for xz in ("X", "Z"):
            self.slider[xz].observe(self.replot_spectra, "value")
        return VBox([Box([self.play, self.step_slider]), self.plots_box])

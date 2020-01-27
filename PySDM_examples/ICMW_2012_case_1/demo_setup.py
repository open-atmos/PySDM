"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

from ipywidgets import IntSlider, FloatSlider, VBox, Checkbox, Accordion
from PySDM_examples.ICMW_2012_case_1.setup import Setup


class DemoSetup(Setup):
    # grid
    ui_nx = IntSlider(value=Setup.grid[0], min=10, max=100, description="nx")
    ui_nz = IntSlider(value=Setup.grid[1], min=10, max=100, description="nz")

    @property
    def grid(self):
        return self.ui_nx.value, self.ui_nz.value

    ui_dt = FloatSlider(value=Setup.dt, min=.1, max=10, description="dt")

    @property
    def dt(self):
        return self.ui_dt.value

    # processes
    ui_processes = [Checkbox(value=Setup.processes[key], description=key) for key in Setup.processes.keys()]

    @property
    def processes(self):
        result = {}
        for checkbox in self.ui_processes:
            result[checkbox.description] = checkbox.value
        return result

    # n_sd_per_gridbox
    ui_sdpg = IntSlider(value=Setup.n_sd_per_gridbox, description="n_sd/gridbox", min=1, max=1024)

    @property
    def n_sd_per_gridbox(self):
        return self.ui_sdpg.value

    # mpdata_options
    ui_mpdata_options = [
        Checkbox(value=Setup.mpdata_fct, description="fct"),
        Checkbox(value=Setup.mpdata_tot, description="tot"),
        Checkbox(value=Setup.mpdata_iga, description="iga"),
        IntSlider(value=Setup.mpdata_iters, description="iters", min=1, max=5)
    ]

    @property
    def mpdata_tot(self):
        for widget in self.ui_mpdata_options:
            if widget.description == 'tot':
                return widget.value
        raise Exception()

    @property
    def mpdata_fct(self):
        for widget in self.ui_mpdata_options:
            if widget.description == 'fct':
                return widget.value
        raise Exception()

    @property
    def mpdata_iga(self):
        for widget in self.ui_mpdata_options:
            if widget.description == 'iga':
                return widget.value
        raise Exception()

    @property
    def mpdata_iters(self):
        for widget in self.ui_mpdata_options:
            if widget.description == 'iters':
                return widget.value
        raise Exception()

    def box(self):
        layout = Accordion(children=[
            VBox([
                self.ui_nx, self.ui_nz,
                self.ui_sdpg,
                self.ui_dt
                ]),
            VBox([*self.ui_processes]),
            VBox([*self.ui_mpdata_options])
        ])
        layout.set_title(0, 'discretisation')
        layout.set_title(1, 'processes')
        layout.set_title(2, 'MPDATA options')
        return layout
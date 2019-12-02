"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

from ipywidgets import IntSlider, FloatSlider, VBox, Checkbox
from examples.ICMW_2012_case_1.setup import Setup


class DemoSetup(Setup):
    # grid
    ui_nx = IntSlider(value=Setup.grid[0], min=10, max=100, description="nx")
    ui_nz = IntSlider(value=Setup.grid[1], min=10, max=100, description="nz")

    @property
    def grid(self):
        return self.ui_nx.value, self.ui_nz.value

    ui_dt = FloatSlider(value=Setup.dt, min=.1, max=1, description="dt")
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

    def box(self):
        return VBox([
            self.ui_nx, self.ui_nz,
            self.ui_sdpg,
            self.ui_dt,
            *self.ui_processes
        ])

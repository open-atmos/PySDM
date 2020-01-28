"""
Created at 02.10.2019

@author: Sylwester Arabas
"""

from ipywidgets import IntSlider, FloatSlider, VBox, Checkbox, Accordion
from PySDM_examples.ICMW_2012_case_1.setup import Setup


class DemoSetup(Setup):
    ui_th_std0 = FloatSlider(description="th0 [K]", value=Setup.th_std0, min=280, max=300)

    @property
    def th_std0(self):
        return self.ui_th_std0.value

    ui_qv0 = FloatSlider(description="qv0 [g/kg]", value=Setup.qv0*1000, min=5, max=10)

    @property
    def qv0(self):
        return self.ui_qv0.value/1000

    ui_p0 = FloatSlider(description="p0 [hPa]", value=Setup.p0/100, min=900, max=1100)

    @property
    def p0(self):
        return self.ui_p0.value*100

    ui_kappa = FloatSlider(description="kappa [1]", value=Setup.kappa, min=0, max=1.5)

    @property
    def kappa(self):
        return self.ui_kappa.value

    ui_w_max = FloatSlider(description="w_max [m/s]", value=Setup.w_max, min=0.1, max=1)

    @property
    def w_max(self):
        return self.ui_w_max.value

    ui_nx = IntSlider(value=Setup.grid[0], min=10, max=100, description="nx")
    ui_nz = IntSlider(value=Setup.grid[1], min=10, max=100, description="nz")

    @property
    def grid(self):
        return self.ui_nx.value, self.ui_nz.value

    ui_dt = FloatSlider(value=Setup.dt, min=.5, max=5, description="dt (Eulerian advection)")

    @property
    def dt(self):
        return self.ui_dt.value

    ui_condensation_dt_max = FloatSlider(value=Setup.condensation_dt_max, min=.05, max=1.01, step=.05, description="dt_max (condensation)")

    @property
    def condensation_dt_max(self):
        return self.ui_condensation_dt_max.value

    ui_processes = [Checkbox(value=Setup.processes[key], description=key) for key in Setup.processes.keys()]

    @property
    def processes(self):
        result = {}
        for checkbox in self.ui_processes:
            result[checkbox.description] = checkbox.value
        return result

    ui_sdpg = IntSlider(value=Setup.n_sd_per_gridbox, description="n_sd/gridbox", min=1, max=100)

    @property
    def n_sd_per_gridbox(self):
        return self.ui_sdpg.value

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
            VBox([self.ui_th_std0, self.ui_qv0, self.ui_p0, self.ui_kappa, self.ui_w_max]),
            VBox([self.ui_nx, self.ui_nz, self.ui_sdpg, self.ui_dt, self.ui_condensation_dt_max]),
            VBox([*self.ui_processes]),
            VBox([*self.ui_mpdata_options])
        ])
        layout.set_title(0, 'parameters')
        layout.set_title(1, 'discretisation')
        layout.set_title(2, 'processes')
        layout.set_title(3, 'MPDATA options')
        return layout
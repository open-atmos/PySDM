"""
Created at 02.10.2019
"""

from ipywidgets import IntSlider, FloatSlider, VBox, Checkbox, Accordion, Dropdown
from PySDM_examples.ICMW_2012_case_1.setup import Setup
import numpy as np


class DemoSetup(Setup):
    ui_th_std0 = FloatSlider(description="$\\theta_0$ [K]", value=Setup.th_std0, min=280, max=300)

    @property
    def th_std0(self):
        return self.ui_th_std0.value

    ui_qv0 = FloatSlider(description="q$_{v0}$ [g/kg]", value=Setup.qv0*1000, min=5, max=10)

    @property
    def qv0(self):
        return self.ui_qv0.value/1000

    ui_p0 = FloatSlider(description="p$_0$ [hPa]", value=Setup.p0/100, min=900, max=1100)

    @property
    def p0(self):
        return self.ui_p0.value*100

    ui_kappa = FloatSlider(description="$\kappa$ [1]", value=Setup.kappa, min=0, max=1.5)

    @property
    def kappa(self):
        return self.ui_kappa.value

    ui_amplitude = FloatSlider(description="$\psi_{_{mx}}$[kg/s/m$^{_2}$]", value=Setup.rho_w_max, min=-1, max=1)

    @property
    def amplitude(self):
        return self.ui_amplitude.value

    ui_nx = IntSlider(value=Setup.grid[0], min=10, max=100, description="nx")
    ui_nz = IntSlider(value=Setup.grid[1], min=10, max=100, description="nz")

    @property
    def grid(self):
        return self.ui_nx.value, self.ui_nz.value

    ui_dt = FloatSlider(value=Setup.dt, min=.5, max=5, description="dt (Eulerian)")

    @property
    def dt(self):
        return self.ui_dt.value

    ui_n_steps = IntSlider(value=Setup.n_steps, min=1800, max=7200, description="# steps")

    @property
    def n_steps(self):
        return self.ui_n_steps.value

    ui_condensation_rtol_x = IntSlider(value=np.log10(Setup.condensation_rtol_thd), min=-9, max=-3, description="log$_{10}$(rtol$_x$)")

    @property
    def condensation_rtol_x(self):
        return 10**self.ui_condensation_rtol_x.value

    ui_condensation_rtol_thd = IntSlider(value=np.log10(Setup.condensation_rtol_thd), min=-9, max=-3, description="log$_{10}$(rtol$_\\theta$)")

    @property
    def condensation_rtol_thd(self):
        return 10**self.ui_condensation_rtol_thd.value

    ui_adaptive = Checkbox(value=Setup.adaptive, description='adaptive timestep')

    @property
    def adaptive(self):
        return self.ui_adaptive.value

    ui_condensation_coord = Dropdown(options=['volume', 'volume logarithm'], value=Setup.condensation_coord, description='condensational variable coordinate')

    @property
    def condensation_coord(self):
        return self.ui_condensation_coord.value

    ui_processes = [Checkbox(value=Setup.processes[key], description=key) for key in Setup.processes.keys()]

    @property
    def processes(self):
        result = {}
        for checkbox in self.ui_processes:
            result[checkbox.description] = checkbox.value
        return result

    # @property
    # def enable_particle_temperatures(self):
    #     return self.ui_ept.value

    ui_sdpg = IntSlider(value=Setup.n_sd_per_gridbox, description="n_sd/gridbox", min=1, max=1000)

    @property
    def n_sd_per_gridbox(self):
        return self.ui_sdpg.value

    fct_description = "MPDATA: flux-corrected transport option"
    tot_description = "MPDATA: third-order terms option"
    iga_description = "MPDATA: infinite gauge option"
    nit_description = "MPDATA: number of iterations (1=UPWIND)"
    ui_mpdata_options = [
        Checkbox(value=Setup.mpdata_fct, description=fct_description),
        Checkbox(value=Setup.mpdata_tot, description=tot_description),
        Checkbox(value=Setup.mpdata_iga, description=iga_description),
        IntSlider(value=Setup.mpdata_iters, description=nit_description, min=1, max=5)
    ]

    @property
    def mpdata_tot(self):
        for widget in self.ui_mpdata_options:
            if widget.description == self.tot_description:
                return widget.value
        raise Exception()

    @property
    def mpdata_fct(self):
        for widget in self.ui_mpdata_options:
            if widget.description == self.fct_description:
                return widget.value
        raise Exception()

    @property
    def mpdata_iga(self):
        for widget in self.ui_mpdata_options:
            if widget.description == self.iga_description:
                return widget.value
        raise Exception()

    @property
    def mpdata_iters(self):
        for widget in self.ui_mpdata_options:
            if widget.description == self.nit_description:
                return widget.value
        raise Exception()

    def box(self):
        layout = Accordion(children=[
            VBox([self.ui_th_std0, self.ui_qv0, self.ui_p0, self.ui_kappa, self.ui_amplitude]),
            VBox([*self.ui_processes]),
            VBox([self.ui_nx, self.ui_nz, self.ui_sdpg, self.ui_dt, self.ui_n_steps,
                  self.ui_condensation_rtol_x, self.ui_condensation_rtol_thd,
                  self.ui_adaptive, self.ui_condensation_coord,
                  *self.ui_mpdata_options]),
        ])
        layout.set_title(0, 'environment parameters')
        layout.set_title(1, 'processes')
        layout.set_title(2, 'discretisation')
        return layout

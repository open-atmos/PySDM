import numba
import numpy as np
from PySDM.backends.numba.numba_helpers import temperature_pressure_RH
from .support import HENRY_CONST, EQUILIBRIUM_CONST, DIFFUSION_CONST, \
    MASS_ACCOMMODATION_COEFFICIENTS, AQUEOUS_COMPOUNDS, GASEOUS_COMPOUNDS, MEMBER, KINETIC_CONST, \
    SPECIFIC_GRAVITY
from PySDM.physics.formulae import mole_fraction_2_mixing_ratio
from ...backends.numba.impl._chemistry_methods import dissolve_env_gases, oxidize, pH2H


class AqueousChemistry:
    def __init__(self, environment_mole_fractions, system_type, n_substep):
        self.environment_mixing_ratios = {}
        for key, compound in GASEOUS_COMPOUNDS.items():
            shape = (1,)  # TODO #440
            self.environment_mixing_ratios[compound] = np.full(
                shape,
                mole_fraction_2_mixing_ratio(environment_mole_fractions[compound], SPECIFIC_GRAVITY[compound])
            )
        self.mesh = None
        self.core = None
        self.env = None

        assert system_type in ('open', 'closed')
        self.system_type = system_type
        assert isinstance(n_substep, int) and n_substep > 0
        self.n_substep = n_substep

        self.kinetic_consts = {}
        self.equilibrium_consts = {}
        self.aqq = {}
        self.do_chemistry_flag = None

    def register(self, builder):
        self.mesh = builder.core.mesh
        self.core = builder.core
        self.env = builder.core.env
        for key in AQUEOUS_COMPOUNDS.keys():
            builder.request_attribute("conc_" + key)

        for key in KINETIC_CONST.keys():
            self.kinetic_consts[key] = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)
        for key in EQUILIBRIUM_CONST.keys():
            self.equilibrium_consts[key] = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)
        for key in DIFFUSION_CONST.keys():
            self.aqq[key] = self.core.Storage.empty(self.core.n_sd, dtype=float)
        self.do_chemistry_flag = self.core.Storage.empty(self.core.n_sd, dtype=bool)

    def recalculate_cell_data(self):
        T = self.env['T'].data
        for i in range(self.core.mesh.n_cell):
            for key in EQUILIBRIUM_CONST.keys():
                self.equilibrium_consts[key].data[i] = EQUILIBRIUM_CONST[key].at(T[i])
            for key in KINETIC_CONST.keys():
                self.kinetic_consts[key].data[i] = KINETIC_CONST[key].at(T[i])

    def recalculate_drop_data(self):
        cell_id = self.core.particles['cell id'].data
        pH = self.core.particles['pH'].data
        for i in range(self.core.n_sd):  # TODO #440?
            H = pH2H(pH[i])
            for key in DIFFUSION_CONST.keys():
                self.aqq[key].data[i] = MEMBER[key](H, self.equilibrium_consts, cell_id[i])

    def __call__(self):
        n_cell = self.mesh.n_cell
        n_threads = 1  # TODO #440
        cell_order = np.arange(n_cell)  # TODO #440
        cell_start_arg = self.core.particles.cell_start.data
        idx = self.core.particles._Particles__idx

        rhod = self.env["rhod"]
        thd = self.env["thd"]
        qv = self.env["qv"]
        prhod = self.env.get_predicted("rhod")

        self.recalculate_cell_data()

        # TODO #435
        for _ in range(self.n_substep):
            self.recalculate_drop_data()

            for thread_id in numba.prange(n_threads):
                for i in range(thread_id, n_cell, n_threads):
                    cell_id = cell_order[i]

                    cell_start = cell_start_arg[cell_id]
                    cell_end = cell_start_arg[cell_id + 1]
                    n_sd_in_cell = cell_end - cell_start
                    if n_sd_in_cell == 0:
                        continue

                    rhod_mean = (prhod[cell_id] + rhod[cell_id]) / 2
                    T, p, RH = temperature_pressure_RH(rhod_mean, thd[cell_id], qv[cell_id])  # TODO #440: this is surely already computed elsewhere!

                    super_droplet_ids = numba.typed.List()
                    for sd_id in idx[cell_start:cell_end]:  # TODO #440: idx?
                        if self.do_chemistry_flag.data[sd_id]:
                            super_droplet_ids.append(sd_id)

                    if len(super_droplet_ids) == 0:
                        continue

                    for key, compound in GASEOUS_COMPOUNDS.items():
                        dissolve_env_gases(
                            super_droplet_ids=super_droplet_ids,
                            mole_amounts=self.core.particles['moles_'+key].data,
                            env_mixing_ratio=self.environment_mixing_ratios[compound][cell_id:cell_id+1],
                            henrysConstant=HENRY_CONST[compound].at(T),  # mol m−3 Pa−1
                            env_p=p,
                            env_T=T,
                            env_rho_d=rhod_mean,
                            dt=self.core.dt/self.n_substep,
                            dv=self.mesh.dv,
                            droplet_volume=self.core.particles["volume"].data,
                            multiplicity=self.core.particles["n"].data,
                            system_type=self.system_type,
                            specific_gravity=SPECIFIC_GRAVITY[compound],
                            alpha=MASS_ACCOMMODATION_COEFFICIENTS[compound],
                            diffusion_constant=DIFFUSION_CONST[compound],
                            ksi=self.aqq[compound].data
                        )
            for key in GASEOUS_COMPOUNDS.keys():
                self.core.particles.attributes[f'moles_{key}'].mark_updated()

            self.recalculate_drop_data()

            oxidize(
                n_sd=self.core.n_sd,
                cell_ids=self.core.particles['cell id'].data,
                do_chemistry_flag=self.do_chemistry_flag.data,
                k0=self.kinetic_consts["k0"].data,
                k1=self.kinetic_consts["k1"].data,
                k2=self.kinetic_consts["k2"].data,
                k3=self.kinetic_consts["k3"].data,
                K_SO2=self.equilibrium_consts["K_SO2"].data,
                K_HSO3=self.equilibrium_consts["K_HSO3"].data,
                aqq_SO2=self.aqq['SO2'].data,
                dt=self.core.dt / self.n_substep,
                # input
                droplet_volume=self.core.particles["volume"].data,
                pH=self.core.particles["pH"].data,
                O3=self.core.particles["conc_O3"].data,
                H2O2=self.core.particles["conc_H2O2"].data,
                S_IV=self.core.particles["conc_S_IV"].data,
                # output
                moles_O3=self.core.particles["moles_O3"].data,
                moles_H2O2=self.core.particles["moles_H2O2"].data,
                moles_S_IV=self.core.particles["moles_S_IV"].data,
                moles_S_VI=self.core.particles["moles_S_VI"].data
            )
            self.core.particles.attributes['moles_S_IV'].mark_updated()
            self.core.particles.attributes['moles_S_VI'].mark_updated()
            self.core.particles.attributes['moles_H2O2'].mark_updated()
            self.core.particles.attributes['moles_O3'].mark_updated()

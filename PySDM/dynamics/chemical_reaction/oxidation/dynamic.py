"""
Created at 18.05.2020

@author: Grzegorz Łazarski
"""


import time
from collections import OrderedDict

import numpy as np
import scipy.optimize as optim

import PySDM.physics.constants as const
from chempy import Substance


from PySDM.physics.constants import si

from .constants import (ROOM_TEMP, M, dry_air_d, gpm_u, depint)
from .reaction_data import (DIFFUSION_CONST, ENVIRONMENT_AMOUNTS,
                            EQUILIBRIUM_CONST, HENRY_CONST, HENRY_PH_DEP, KINETIC_CONST)
from .support import (henry_factory, hydrogen_conc_factory, oxidation_factory,
                      amount_to_dry_v, calc_ionic_strength, dry_v_to_amount)

PROPERTIES = ["n", "dry volume", "volume"]


def default_init(dry_v, wet_v):
    return np.zeros_like(dry_v)


def compound_init(dry_v, wet_v):
    return dry_v_to_amount(dry_v)


COMPOUNDS = OrderedDict({
    "SO2": default_init,
    "O3": default_init,
    "H2O2": default_init,
    "CO2": default_init,
    "HNO3": default_init,
    "NH3": compound_init,
    "HSO4m": compound_init,
    # "Hp": lambda d, w: default_init(d, w) + 1e-7 * M * w,
    "Hp": compound_init,
})


class SDChemistry:

    OXIDATION = ["Hp", "O3", "SO2", "H2O2", "HSO4m"]

    def __init__(self, compounds, particle):

        self.particles = particle

        self._indices = list(compounds)
        self._map = {v: i for i, v in enumerate(self._indices)}

        self.henry_processes = [
            henry_factory(
                *DIFFUSION_CONST[k],
                Substance.from_formula(k).mass * gpm_u,
                v,
                HENRY_PH_DEP[k])
            for k, v in HENRY_CONST.items() if k in self._indices
        ]

        self.hindex = self._map["Hp"]
        self.gas_indices = np.array([self._map[k]
                                     for k, _ in HENRY_CONST.items()])
        self.oxidation_indices = [self._map[k] for k in SDChemistry.OXIDATION]

        # Properly initialized in update_conditions
        self.ideal_gas_volume = None
        self.update_conditions(self.particles.environment["T"],
                               self.particles.environment["p"])

        self.environment_indices = dict()
        i = 0
        for k in self.gas_indices:
            if self._indices[k] in ENVIRONMENT_AMOUNTS:
                self.environment_indices[self._indices[k]] = i
                i += 1

        # Converting from volume ppb to molarity (M)
        self.environment = np.array([
                    ENVIRONMENT_AMOUNTS[name]
                        / self.ideal_gas_volume
                        for name, idex in self.environment_indices.items()
                ])

    def dict(self, amounts):
        return {k: v for k, v in zip(self._indices, amounts)}

    def update_conditions(self, T=const.T_STP, p=const.p_STP):
        self.ideal_gas_volume = const.R_str * T / p
        self.eq_const_temp = {k: v.at(T)
                              for k, v in EQUILIBRIUM_CONST.items()}

    def dissolve_env_gases(self, amounts, V_w, n, *, dt, steps=1, T=ROOM_TEMP):
        hconc = amounts[self.hindex]
        for henry, gasi, envi in zip(self.henry_processes,
                                     self.gas_indices,
                                     self.environment_indices.values()):
            A = amounts[gasi]
            c = self.environment[envi]
            for i in range(steps):
                A = henry(A, c, T=T, V_w=V_w, H=hconc, dt=dt)

                # Don't accumulate to avoid numerical errors
                # We multiply by n, since all the droplets suck in the gases
                c = (self.environment[envi]
                     - (A - amounts[gasi]) * n * V_w / self.particles.environment.dv)

                # Ensure we do not take too much
                if A < 0:
                    print(f"Borrowed gas: {self._indices[gasi]}")
                    A = 0
                if c < 0:
                    print(f"Borrowed gas: {self._indices[gasi]}")
                    c = 0
            amounts[gasi] = A
            self.environment[envi] = c

        return amounts

    def equilibriate_ph(self, amounts):
        f = hydrogen_conc_factory(**self.dict(amounts), **self.eq_const_temp)
        result = optim.root_scalar(
            f, x0=depint(1e-10 * M), x1=depint(10 * M))
        if not result.converged:
            raise ValueError("Hp concentration failed to find a root.")
        amounts[self.hindex] = result.root
        return amounts

    def oxidize(self, amounts, T=ROOM_TEMP, *, dt, steps):
        ks = {k: v.at(T) for k, v in KINETIC_CONST.items()}
        fraw = oxidation_factory(
            **ks, **{k: EQUILIBRIUM_CONST[k].at(T)
                     for k in ["K_SO2", "K_HSO3"]})

        new_amounts = amounts[self.oxidation_indices]
        for i in range(steps):
            new_amounts += np.array(fraw(new_amounts)) * dt

            # Ensure we do not take too much
            new_amounts[new_amounts < 0] = 0

        amounts[self.oxidation_indices] = new_amounts
        return amounts

    def skip_chemistry(self, amounts):
        stren = calc_ionic_strength(**self.dict(amounts),
                                    **self.eq_const_temp)
        return stren >= 0.02 * M


class ChemicalReaction:
    def __init__(self, particles_builder, *,
                 dissolve_steps=5, oxidize_steps=5, oxidize_timestep=None, dissolve_timestep=None):
        self.particles = particles_builder.particles

        self.dissolve_steps = dissolve_steps
        self.oxidize_steps = oxidize_steps

        if dissolve_timestep is None:
            self.dissolve_timestep = self.particles.dt / 2
        else:
            self.dissolve_timestep = dissolve_timestep

        if oxidize_timestep is None:
            self.oxidize_timestep = self.particles.dt / 2
        else:
            self.oxidize_timestep = oxidize_timestep

        self.sdchem = SDChemistry(COMPOUNDS.keys(), self.particles)
        # 1 * si.kg / (1.2250 * si.kg / si.m ** 3))

    def __call__(self):
        t0 = time.perf_counter_ns()

        data = [self.particles.state[x]
                for x in PROPERTIES]
        amounts = [self.particles.state[x] for x in COMPOUNDS.keys()]

        skipped = 0

        self.sdchem.update_conditions(self.particles.environment["T"],
                                      self.particles.environment["p"])

        for i, n, dry, wet, amount in zip(
                self.particles.state._State__idx, *data, zip(*amounts)):

            concentrations = np.array(amount) / wet

            if self.sdchem.skip_chemistry(concentrations):
                skipped += 1
            else:

                concentrations = self.sdchem.dissolve_env_gases(
                    concentrations,
                    wet,
                    n,
                    dt=self.dissolve_timestep / self.dissolve_steps,
                    steps=self.dissolve_steps)

                if concentrations[self.sdchem.hindex]:
                    concentrations[self.sdchem.hindex] = 1e-7 * M

                concentrations = self.sdchem.equilibriate_ph(concentrations)

                if concentrations[self.sdchem.hindex]:
                    concentrations[self.sdchem.hindex] = 1e-7 * M

                concentrations = self.sdchem.oxidize(
                    concentrations,
                    dt=self.oxidize_timestep / self.oxidize_steps,
                    steps=self.oxidize_steps)

                if concentrations[self.sdchem.hindex]:
                    concentrations[self.sdchem.hindex] = 1e-7 * M

            dict_amounts = self.sdchem.dict(concentrations)
            new_dry_conc = min(dict_amounts["NH3"], dict_amounts["HSO4m"])
            new_dry_v = amount_to_dry_v(new_dry_conc * wet)

            # assert(new_dry_v >= dry)

            # TODO verify indexing is correct
            self.particles.state["dry volume"][i] = new_dry_v
            for k, v in dict_amounts.items():
                self.particles.state[k][i] = v * wet

        t1 = time.perf_counter_ns()
        print(f"Chemistry total took {(t1 - t0) * si.ns :0.3f}s,"
              f" skipped {skipped}/{self.particles.n_sd}")
        # print(f"Ammonium conc: {np.mean(self.particles.state['NH3'])}")

    @staticmethod
    def get_starting_amounts(dry_v, wet_v):
        return {k: v(dry_v, wet_v) for k, v in COMPOUNDS.items()}

import time
from collections import OrderedDict

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optim

from chempy import Substance
from PySDM.dynamics.chemical_reaction.oxidation.constants import (ROOM_TEMP, M,
                                                                  depint,
                                                                  gpm_u)
from PySDM.dynamics.chemical_reaction.oxidation.reaction_data import (
    DIFFUSION_CONST, ENVIRONMENT_AMOUNTS, EQUILIBRIUM_CONST, HENRY_CONST,
    KINETIC_CONST)
from PySDM.dynamics.chemical_reaction.oxidation.support import (
    amount_to_dry_v, dry_v_to_amount)
from PySDM.dynamics.chemical_reaction.products.particles_dry_volume_spectrum import \
    ParticlesDryVolumeSpectrum
from PySDM.physics.constants import si

from .support import henry_factory, hydrogen_conc_factory, oxidation_factory

PROPERTIES = ["dry volume", "volume"]


def default_init(dry_v):
    return np.zeros_like(dry_v)


def compound_init(dry_v):
    return dry_v_to_amount(dry_v)


COMPOUNDS = OrderedDict({
    "SO2": default_init,
    "O3": default_init,
    "H2O2": default_init,
    "CO2": default_init,
    "HNO3": default_init,
    "NH3": compound_init,
    "HSO4m": compound_init,
    "H+": lambda x: np.full_like(x, 0)
})


class SDChemistry:

    OXIDATION = ["H+", "O3", "SO2", "H2O2", "HSO4m"]

    def __init__(self, compounds):

        self._indices = list(compounds)
        self._map = {v: i for i, v in enumerate(self._indices)}

        self.henry_processes = [
            henry_factory(
                *DIFFUSION_CONST[k],
                Substance.from_formula(k).mass * gpm_u,
                v)
            for k, v in HENRY_CONST.items() if k in self._indices
        ]

        # TODO: dynamic envorinemnt concentrations
        self.environment = [ENVIRONMENT_AMOUNTS[k] * M
                            for k in self._indices if k in ENVIRONMENT_AMOUNTS]

        self.hindex = self._map["H+"]
        self.gas_indices = np.array([self._map[k]
                                     for k, _ in HENRY_CONST.items()])
        self.oxidation_indices = [self._map[k] for k in SDChemistry.OXIDATION]

        self.depint_equilib = {k: depint(v.K)
                               for k, v in EQUILIBRIUM_CONST.items()}

    def dict(self, amounts):
        return {k: v for k, v in zip(self._indices, amounts)}

    def dissolve_env_gases(self, amounts, V_w, T=ROOM_TEMP, *, t_interval):
        environment = self.environment

        def f(As, t, T, V_w):
            return [depint(
                proc(A * M, T=T, cinf=env, V_w=V_w))
                for proc, A, env in zip(self.henry_processes, As, environment)]
        raw_result = integrate.odeint(
            f, amounts[self.gas_indices], t_interval, args=(T, V_w))[-1]

        sanit_indices = np.full_like(raw_result, True).astype(np.int32)
        # sanit_indices = raw_result > amounts[self.gas_indices]
        amounts[self.gas_indices[sanit_indices]] = raw_result[sanit_indices]

        return amounts

    def equilibriate_ph(self, amounts):
        f = hydrogen_conc_factory(
            **self.dict(amounts), **self.depint_equilib)
        result = optim.root_scalar(
            f, x0=depint(1e-7 * M), x1=depint(10 * M))
        if not result.converged:
            raise ValueError("H+ concentration failed to find a root.")
        amounts[self.hindex] = result.root
        return amounts

    def oxidize(self, amounts, T=ROOM_TEMP, *, t_interval):
        ks = {k: depint(v.at(T)) for k, v in KINETIC_CONST.items()}
        fraw = oxidation_factory(
            **ks, **self.depint_equilib)

        def f(x, t):
            return fraw(x)

        raw_res = integrate.odeint(
            f, amounts[self.oxidation_indices], t_interval)[-1]
        raw_res[raw_res < 0] = 0
        amounts[self.oxidation_indices] = raw_res
        return amounts


class ChemicalReaction:
    def __init__(self, particles_builder, *, dissolve_steps=10, oxidize_steps=10):
        self.particles = particles_builder.particles

        self.dissolve_t = np.linspace(
            0, self.particles.dt/2, num=dissolve_steps)
        self.oxidize_t = np.linspace(0, self.particles.dt/2, num=oxidize_steps)

        self.sdchem = SDChemistry(COMPOUNDS.keys())

        self.products = [ParticlesDryVolumeSpectrum(self.particles), ]

    def __call__(self):
        t0 = time.perf_counter_ns()
        data = [self.particles.state.get_backend_storage(x)
                for x in PROPERTIES]
        amounts = [self.particles.state.get_backend_storage(
            x) for x in COMPOUNDS.keys()]

        # print(len(list(zip(*amounts))))
        for i, (dry, wet, amount) in enumerate(zip(*data, zip(*amounts))):
            amount = np.array(amount) / wet
            # print(dry, wet, amount)
            # print(self.sdchem.dict(amount))
            amount = self.sdchem.dissolve_env_gases(
                amount, wet, t_interval=self.dissolve_t)
            # print(self.sdchem.dict(amount))
            amount = self.sdchem.equilibriate_ph(amount)
            # print(amount[self.sdchem.hindex], "- pH:",
            #       - np.log10(amount[self.sdchem.hindex]))
            # print(*zip(SDChemistry.OXIDATION,
            #            amount[self.sdchem.oxidation_indices]))
            amount = self.sdchem.oxidize(amount, t_interval=self.oxidize_t)
            # print(*zip(SDChemistry.OXIDATION,
            #            amount[self.sdchem.oxidation_indices]))

            # print({k: amount[i] for i, k in enumerate(self.sdchem._indices)})

            if i == 0:
                print(
                    f"Was @ {i}: {data[0][i]}")

            dict_amounts = self.sdchem.dict(amount)
            new_dry_conc = min(dict_amounts["NH3"], dict_amounts["HSO4m"])
            new_dry_v = amount_to_dry_v(new_dry_conc * wet)

            data[0][i] = new_dry_v
            for j, v in enumerate(amount):
                amounts[j][i] = v * wet

            if i == 0:
                print(
                    f"Calculated @ {i}: {data[0][i]}")
                print(
                    f"Chemistry step took {(time.perf_counter_ns() - t0) * si.ns}s")
        print(f"Chemistry total took {(time.perf_counter_ns() - t0) * si.ns}s")

    @staticmethod
    def get_starting_amounts(dry_v):
        return {k: v(dry_v) for k, v in COMPOUNDS.items()}

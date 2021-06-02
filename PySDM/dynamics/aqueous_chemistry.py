"""
Hoppel-gap resolving aqueous-phase chemistry (incl. SO2 oxidation)
"""
import numpy as np
from PySDM.physics.aqueous_chemistry.support import DIFFUSION_CONST, AQUEOUS_COMPOUNDS, \
    GASEOUS_COMPOUNDS, SPECIFIC_GRAVITY, M


default_pH_min = -1.
default_pH_max = 14.
default_pH_rtol = 1e-6
default_ionic_strength_threshold = 0.02 * M


class AqueousChemistry:
    def __init__(self, environment_mole_fractions, system_type, n_substep, dry_rho, dry_molar_mass,
                 ionic_strength_threshold=default_ionic_strength_threshold,
                 pH_H_min=None,
                 pH_H_max=None,
                 pH_rtol=default_pH_rtol):
        self.environment_mole_fractions = environment_mole_fractions
        self.environment_mixing_ratios = {}
        self.core = None

        assert system_type in ('open', 'closed')
        self.system_type = system_type
        assert isinstance(n_substep, int) and n_substep > 0
        self.n_substep = n_substep
        self.dry_rho = dry_rho
        self.dry_molar_mass = dry_molar_mass
        self.ionic_strength_threshold=ionic_strength_threshold
        self.pH_H_max = pH_H_max
        self.pH_H_min = pH_H_min
        self.pH_rtol = pH_rtol

        self.kinetic_consts = {}
        self.equilibrium_consts = {}
        self.dissociation_factors = {}
        self.do_chemistry_flag = None

    def register(self, builder):
        self.core = builder.core

        for key, compound in GASEOUS_COMPOUNDS.items():
            shape = (1,)
            self.environment_mixing_ratios[compound] = np.full(
                shape,
                self.core.formulae.trivia.mole_fraction_2_mixing_ratio(
                    self.environment_mole_fractions[compound], SPECIFIC_GRAVITY[compound])
            )
        self.environment_mole_fractions = None

        if self.pH_H_max is None:
            self.pH_H_max = self.core.formulae.trivia.pH2H(default_pH_min)
        if self.pH_H_min is None:
            self.pH_H_min = self.core.formulae.trivia.pH2H(default_pH_max)

        for key in AQUEOUS_COMPOUNDS.keys():
            builder.request_attribute("conc_" + key)

        for key in self.core.backend.KINETIC_CONST.KINETIC_CONST.keys():
            self.kinetic_consts[key] = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)
        for key in self.core.backend.EQUILIBRIUM_CONST.EQUILIBRIUM_CONST.keys():
            self.equilibrium_consts[key] = self.core.Storage.empty(self.core.mesh.n_cell, dtype=float)
        for key in DIFFUSION_CONST.keys():
            self.dissociation_factors[key] = self.core.Storage.empty(self.core.n_sd, dtype=float)
        self.do_chemistry_flag = self.core.Storage.empty(self.core.n_sd, dtype=bool)

    def __call__(self):
        self.core.particles.chem_recalculate_cell_data(
            equilibrium_consts=self.equilibrium_consts,
            kinetic_consts=self.kinetic_consts
        )
        for _ in range(self.n_substep):
            self.core.particles.chem_recalculate_drop_data(
                equilibrium_consts=self.equilibrium_consts,
                dissociation_factors=self.dissociation_factors
            )
            self.core.particles.dissolution(
                gaseous_compounds=GASEOUS_COMPOUNDS,
                system_type=self.system_type,
                dissociation_factors=self.dissociation_factors,
                environment_mixing_ratios=self.environment_mixing_ratios,
                dt=self.core.dt / self.n_substep,
                do_chemistry_flag=self.do_chemistry_flag
            )
            self.core.particles.chem_recalculate_drop_data(
                equilibrium_consts=self.equilibrium_consts,
                dissociation_factors=self.dissociation_factors
            )
            self.core.particles.oxidation(
                kinetic_consts=self.kinetic_consts,
                equilibrium_consts=self.equilibrium_consts,
                dissociation_factors=self.dissociation_factors,
                do_chemistry_flag=self.do_chemistry_flag,
                dt=self.core.dt / self.n_substep
            )

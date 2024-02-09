"""
Hoppel-gap resolving aqueous-phase chemistry (incl. SO2 oxidation)
"""

from collections import namedtuple

import numpy as np

from PySDM.dynamics.impl.chemistry_utils import (
    AQUEOUS_COMPOUNDS,
    DIFFUSION_CONST,
    GASEOUS_COMPOUNDS,
    M,
    SpecificGravities,
)

DEFAULTS = namedtuple("_", ("pH_min", "pH_max", "pH_rtol", "ionic_strength_threshold"))(
    pH_min=-1.0, pH_max=14.0, pH_rtol=1e-6, ionic_strength_threshold=0.02 * M
)


class AqueousChemistry:  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        *,
        environment_mole_fractions,
        system_type,
        n_substep,
        dry_rho,
        dry_molar_mass,
        ionic_strength_threshold=DEFAULTS.ionic_strength_threshold,
        pH_H_min=None,
        pH_H_max=None,
        pH_rtol=DEFAULTS.pH_rtol,
    ):
        self.environment_mole_fractions = environment_mole_fractions
        self.environment_mixing_ratios = {}
        self.particulator = None

        assert system_type in ("open", "closed")
        self.system_type = system_type
        assert isinstance(n_substep, int) and n_substep > 0
        self.n_substep = n_substep
        self.dry_rho = dry_rho
        self.dry_molar_mass = dry_molar_mass
        self.ionic_strength_threshold = ionic_strength_threshold
        self.pH_H_max = pH_H_max
        self.pH_H_min = pH_H_min
        self.pH_rtol = pH_rtol

        self.kinetic_consts = {}
        self.equilibrium_consts = {}
        self.dissociation_factors = {}
        self.do_chemistry_flag = None
        self.specific_gravities = None

    def register(self, builder):
        self.particulator = builder.particulator
        self.specific_gravities = SpecificGravities(
            self.particulator.formulae.constants
        )

        for key, compound in GASEOUS_COMPOUNDS.items():
            shape = (1,)
            self.environment_mixing_ratios[compound] = np.full(
                shape,
                self.particulator.formulae.trivia.mole_fraction_2_mixing_ratio(
                    self.environment_mole_fractions[compound],
                    self.specific_gravities[compound],
                ),
            )
        self.environment_mole_fractions = None

        if self.pH_H_max is None:
            self.pH_H_max = self.particulator.formulae.trivia.pH2H(DEFAULTS.pH_min)
        if self.pH_H_min is None:
            self.pH_H_min = self.particulator.formulae.trivia.pH2H(DEFAULTS.pH_max)

        for key in AQUEOUS_COMPOUNDS:
            builder.request_attribute("conc_" + key)
        builder.request_attribute("pH")

        for key in self.particulator.backend.KINETIC_CONST.KINETIC_CONST:
            self.kinetic_consts[key] = self.particulator.Storage.empty(
                self.particulator.mesh.n_cell, dtype=float
            )
        for key in self.particulator.backend.EQUILIBRIUM_CONST.EQUILIBRIUM_CONST:
            self.equilibrium_consts[key] = self.particulator.Storage.empty(
                self.particulator.mesh.n_cell, dtype=float
            )
        for key in DIFFUSION_CONST:
            self.dissociation_factors[key] = self.particulator.Storage.empty(
                self.particulator.n_sd, dtype=float
            )
        self.do_chemistry_flag = self.particulator.Storage.empty(
            self.particulator.n_sd, dtype=bool
        )

    def __call__(self):
        self.particulator.chem_recalculate_cell_data(
            equilibrium_consts=self.equilibrium_consts,
            kinetic_consts=self.kinetic_consts,
        )
        for _ in range(self.n_substep):
            self.particulator.chem_recalculate_drop_data(
                equilibrium_consts=self.equilibrium_consts,
                dissociation_factors=self.dissociation_factors,
            )
            self.particulator.dissolution(
                gaseous_compounds=GASEOUS_COMPOUNDS,
                system_type=self.system_type,
                dissociation_factors=self.dissociation_factors,
                environment_mixing_ratios=self.environment_mixing_ratios,
                timestep=self.particulator.dt / self.n_substep,
                do_chemistry_flag=self.do_chemistry_flag,
            )
            self.particulator.chem_recalculate_drop_data(
                equilibrium_consts=self.equilibrium_consts,
                dissociation_factors=self.dissociation_factors,
            )
            self.particulator.oxidation(
                kinetic_consts=self.kinetic_consts,
                equilibrium_consts=self.equilibrium_consts,
                dissociation_factors=self.dissociation_factors,
                do_chemistry_flag=self.do_chemistry_flag,
                timestep=self.particulator.dt / self.n_substep,
            )

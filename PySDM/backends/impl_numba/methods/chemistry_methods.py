"""
CPU implementation of backend methods for aqueous chemistry
"""

from collections import namedtuple

import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf
from PySDM.backends.impl_numba.toms748 import toms748_solve
from PySDM.dynamics.impl.chemistry_utils import (
    DIFFUSION_CONST,
    DISSOCIATION_FACTORS,
    GASEOUS_COMPOUNDS,
    MASS_ACCOMMODATION_COEFFICIENTS,
    EquilibriumConsts,
    HenryConsts,
    KineticConsts,
    SpecificGravities,
    k4,
)
from PySDM.physics.constants import K_H2O

_MAX_ITER_QUITE_CLOSE = 8
_MAX_ITER_DEFAULT = 32
_REALY_CLOSE_THRESHOLD = 1e-6
_QUITE_CLOSE_THRESHOLD = 1
_QUITE_CLOSE_MULTIPLIER = 2

_K = namedtuple("_K", ("NH3", "SO2", "HSO3", "HSO4", "HCO3", "CO2", "HNO3"))
_conc = namedtuple("_conc", ("N_mIII", "N_V", "C_IV", "S_IV", "S_VI"))


class ChemistryMethods(BackendMethods):
    def __init__(self):
        BackendMethods.__init__(self)
        self.HENRY_CONST = HenryConsts(self.formulae)
        self.KINETIC_CONST = KineticConsts(self.formulae)
        self.EQUILIBRIUM_CONST = EquilibriumConsts(self.formulae)
        self.specific_gravities = SpecificGravities(self.formulae.constants)

    def dissolution(  # pylint:disable=too-many-locals
        self,
        *,
        n_cell,
        n_threads,
        cell_order,
        cell_start_arg,
        idx,
        do_chemistry_flag,
        mole_amounts,
        env_mixing_ratio,
        env_T,
        env_p,
        env_rho_d,
        dissociation_factors,
        timestep,
        dv,
        system_type,
        droplet_volume,
        multiplicity,
    ):
        assert n_cell == 1
        assert n_threads == 1
        for i in range(n_cell):
            cell_id = cell_order[i]

            cell_start = cell_start_arg[cell_id]
            cell_end = cell_start_arg[cell_id + 1]
            n_sd_in_cell = cell_end - cell_start
            if n_sd_in_cell == 0:
                continue

            super_droplet_ids = numba.typed.List()
            for sd_id in idx[cell_start:cell_end]:
                if do_chemistry_flag.data[sd_id]:
                    super_droplet_ids.append(sd_id)

            if len(super_droplet_ids) == 0:
                return

            for key, compound in GASEOUS_COMPOUNDS.items():
                ChemistryMethods.dissolution_body(
                    super_droplet_ids=super_droplet_ids,
                    mole_amounts=mole_amounts[key].data,
                    env_mixing_ratio=env_mixing_ratio[compound][cell_id : cell_id + 1],
                    henrysConstant=self.HENRY_CONST.HENRY_CONST[compound].at(
                        env_T[cell_id]
                    ),
                    env_p=env_p[cell_id],
                    env_T=env_T[cell_id],
                    env_rho_d=env_rho_d[cell_id],
                    timestep=timestep,
                    dv=dv,
                    droplet_volume=droplet_volume.data,
                    multiplicity=multiplicity.data,
                    system_type=system_type,
                    specific_gravity=self.specific_gravities[compound],
                    alpha=MASS_ACCOMMODATION_COEFFICIENTS[compound],
                    diffusion_const=DIFFUSION_CONST[compound],
                    dissociation_factor=dissociation_factors[compound].data,
                    radius=self.formulae.trivia.radius,
                    const=self.formulae.constants,
                )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
    def dissolution_body(  # pylint: disable=too-many-locals
        *,
        super_droplet_ids,
        mole_amounts,
        env_mixing_ratio,
        henrysConstant,
        env_p,
        env_T,
        env_rho_d,
        timestep,
        dv,
        droplet_volume,
        multiplicity,
        system_type,
        specific_gravity,
        alpha,
        diffusion_const,
        dissociation_factor,
        radius,
        const,
    ):
        mole_amount_taken = 0
        for i in super_droplet_ids:
            Mc = specific_gravity * const.Md
            Rc = const.R_str / Mc
            cinf = env_p / env_T / (const.Rd / env_mixing_ratio[0] + Rc) / Mc
            r_w = radius(volume=droplet_volume[i])
            v_avg = np.sqrt(8 * const.R_str * env_T / (np.pi * Mc))
            dt_over_scale = timestep / (
                4 * r_w / (3 * v_avg * alpha) + r_w**2 / (3 * diffusion_const)
            )
            A_old = mole_amounts[i] / droplet_volume[i]
            H_eff = henrysConstant * dissociation_factor[i]
            A_new = (A_old + dt_over_scale * cinf) / (
                1 + dt_over_scale / H_eff / const.R_str / env_T
            )
            new_mole_amount_per_real_droplet = A_new * droplet_volume[i]
            assert new_mole_amount_per_real_droplet >= 0

            mole_amount_taken += multiplicity[i] * (
                new_mole_amount_per_real_droplet - mole_amounts[i]
            )
            mole_amounts[i] = new_mole_amount_per_real_droplet
        delta_mr = mole_amount_taken * specific_gravity * const.Md / (dv * env_rho_d)
        assert delta_mr <= env_mixing_ratio
        if system_type == "closed":
            env_mixing_ratio -= delta_mr

    def oxidation(  # pylint: disable=too-many-locals
        self,
        *,
        n_sd,
        cell_ids,
        do_chemistry_flag,
        k0,
        k1,
        k2,
        k3,
        K_SO2,
        K_HSO3,
        timestep,
        droplet_volume,
        pH,
        dissociation_factor_SO2,
        # output
        moles_O3,
        moles_H2O2,
        moles_S_IV,
        moles_S_VI,
    ):
        ChemistryMethods.oxidation_body(
            n_sd=n_sd,
            cell_ids=cell_ids.data,
            do_chemistry_flag=do_chemistry_flag.data,
            explicit_euler=self.formulae.trivia.explicit_euler,
            pH2H=self.formulae.trivia.pH2H,
            k0=k0.data,
            k1=k1.data,
            k2=k2.data,
            k3=k3.data,
            K_SO2=K_SO2.data,
            K_HSO3=K_HSO3.data,
            timestep=timestep,
            droplet_volume=droplet_volume.data,
            pH=pH.data,
            dissociation_factor_SO2=dissociation_factor_SO2.data,
            # output
            moles_O3=moles_O3.data,
            moles_H2O2=moles_H2O2.data,
            moles_S_IV=moles_S_IV.data,
            moles_S_VI=moles_S_VI.data,
        )

    @staticmethod
    @numba.njit(**conf.JIT_FLAGS)
    def oxidation_body(  # pylint: disable=too-many-locals
        *,
        n_sd,
        cell_ids,
        do_chemistry_flag,
        explicit_euler,
        pH2H,
        k0,
        k1,
        k2,
        k3,
        K_SO2,
        K_HSO3,
        timestep,
        droplet_volume,
        pH,
        dissociation_factor_SO2,
        # output
        moles_O3,
        moles_H2O2,
        moles_S_IV,
        moles_S_VI,
    ):
        for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
            if not do_chemistry_flag[i]:
                continue

            cid = cell_ids[i]
            H = pH2H(pH[i])
            SO2aq = moles_S_IV[i] / droplet_volume[i] / dissociation_factor_SO2[i]

            # NB: This might not be entirely correct
            # https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JD092iD04p04171
            # https://www.atmos-chem-phys.net/16/1693/2016/acp-16-1693-2016.pdf

            ozone = (
                (
                    k0[cid]
                    + (k1[cid] * K_SO2[cid] / H)
                    + (k2[cid] * K_SO2[cid] * K_HSO3[cid] / H**2)
                )
                * (moles_O3[i] / droplet_volume[i])
                * SO2aq
            )
            peroxide = (
                k3[cid]
                * K_SO2[cid]
                / (1 + k4 * H)
                * (moles_H2O2[i] / droplet_volume[i])
                * SO2aq
            )
            dt_times_volume = timestep * droplet_volume[i]

            dconc_dt_O3 = -ozone
            dconc_dt_S_IV = -(ozone + peroxide)
            dconc_dt_H2O2 = -peroxide
            dconc_dt_S_VI = ozone + peroxide

            if (
                moles_O3[i] + dconc_dt_O3 * dt_times_volume < 0
                or moles_S_IV[i] + dconc_dt_S_IV * dt_times_volume < 0
                or moles_S_VI[i] + dconc_dt_S_VI * dt_times_volume < 0
                or moles_H2O2[i] + dconc_dt_H2O2 * dt_times_volume < 0
            ):
                continue

            moles_O3[i] = explicit_euler(moles_O3[i], dt_times_volume, dconc_dt_O3)
            moles_S_IV[i] = explicit_euler(
                moles_S_IV[i], dt_times_volume, dconc_dt_S_IV
            )
            moles_S_VI[i] = explicit_euler(
                moles_S_VI[i], dt_times_volume, dconc_dt_S_VI
            )
            moles_H2O2[i] = explicit_euler(
                moles_H2O2[i], dt_times_volume, dconc_dt_H2O2
            )

    def chem_recalculate_drop_data(
        self, dissociation_factors, equilibrium_consts, cell_id, pH
    ):
        for i in range(len(pH)):
            H = self.formulae.trivia.pH2H(pH.data[i])
            for key in DIFFUSION_CONST:
                dissociation_factors[key].data[i] = DISSOCIATION_FACTORS[key](
                    H, equilibrium_consts, cell_id.data[i]
                )

    def chem_recalculate_cell_data(
        self, equilibrium_consts, kinetic_consts, temperature
    ):
        for i in range(len(temperature)):
            for key in equilibrium_consts:
                equilibrium_consts[key].data[i] = (
                    self.EQUILIBRIUM_CONST.EQUILIBRIUM_CONST[key].at(
                        temperature.data[i]
                    )
                )
            for key in kinetic_consts:
                kinetic_consts[key].data[i] = self.KINETIC_CONST.KINETIC_CONST[key].at(
                    temperature.data[i]
                )

    def equilibrate_H(
        self,
        *,
        equilibrium_consts,
        cell_id,
        conc,
        do_chemistry_flag,
        pH,
        H_min,
        H_max,
        ionic_strength_threshold,
        rtol,
    ):
        ChemistryMethods.equilibrate_H_body(
            within_tolerance=self.formulae.trivia.within_tolerance,
            pH2H=self.formulae.trivia.pH2H,
            H2pH=self.formulae.trivia.H2pH,
            cell_id=cell_id.data,
            conc=_conc(
                N_mIII=conc.N_mIII.data,
                N_V=conc.N_V.data,
                C_IV=conc.C_IV.data,
                S_IV=conc.S_IV.data,
                S_VI=conc.S_VI.data,
            ),
            K=_K(
                NH3=equilibrium_consts["K_NH3"].data,
                SO2=equilibrium_consts["K_SO2"].data,
                HSO3=equilibrium_consts["K_HSO3"].data,
                HSO4=equilibrium_consts["K_HSO4"].data,
                HCO3=equilibrium_consts["K_HCO3"].data,
                CO2=equilibrium_consts["K_CO2"].data,
                HNO3=equilibrium_consts["K_HNO3"].data,
            ),
            # output
            do_chemistry_flag=do_chemistry_flag.data,
            pH=pH.data,
            # params
            H_min=H_min,
            H_max=H_max,
            ionic_strength_threshold=ionic_strength_threshold,
            rtol=rtol,
        )

    @staticmethod
    @numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False, "cache": False}})
    def equilibrate_H_body(  # pylint: disable=too-many-arguments,too-many-locals
        within_tolerance,
        pH2H,
        H2pH,
        cell_id,
        conc,
        K,
        do_chemistry_flag,
        pH,
        # params
        H_min,
        H_max,
        ionic_strength_threshold,
        rtol,
    ):
        for i, pH_i in enumerate(pH):
            cid = cell_id[i]
            args = (
                _conc(
                    N_mIII=conc.N_mIII[i],
                    N_V=conc.N_V[i],
                    C_IV=conc.C_IV[i],
                    S_IV=conc.S_IV[i],
                    S_VI=conc.S_VI[i],
                ),
                _K(
                    NH3=K.NH3[cid],
                    SO2=K.SO2[cid],
                    HSO3=K.HSO3[cid],
                    HSO4=K.HSO4[cid],
                    HCO3=K.HCO3[cid],
                    CO2=K.CO2[cid],
                    HNO3=K.HNO3[cid],
                ),
            )
            a = pH2H(pH_i)
            fa = acidity_minfun(a, *args)
            if abs(fa) < _REALY_CLOSE_THRESHOLD:
                continue
            b = np.nan
            fb = np.nan
            use_default_range = False
            if abs(fa) < _QUITE_CLOSE_THRESHOLD:
                b = a * _QUITE_CLOSE_MULTIPLIER
                fb = acidity_minfun(b, *args)
                if fa * fb > 0:
                    b = a
                    fb = fa
                    a = b / _QUITE_CLOSE_MULTIPLIER / _QUITE_CLOSE_MULTIPLIER
                    fa = acidity_minfun(a, *args)
                    if fa * fb > 0:
                        use_default_range = True
            else:
                use_default_range = True
            if use_default_range:
                a = H_min
                b = H_max
                fa = acidity_minfun(a, *args)
                fb = acidity_minfun(b, *args)
                max_iter = _MAX_ITER_DEFAULT
            else:
                max_iter = _MAX_ITER_QUITE_CLOSE
            H, _iters_taken = toms748_solve(
                acidity_minfun,
                args,
                a,
                b,
                fa,
                fb,
                rtol=rtol,
                max_iter=max_iter,
                within_tolerance=within_tolerance,
            )
            assert _iters_taken != max_iter
            pH[i] = H2pH(H)
            ionic_strength = calc_ionic_strength(H, *args)
            do_chemistry_flag[i] = ionic_strength <= ionic_strength_threshold


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def calc_ionic_strength(H, conc, K):
    # Directly adapted
    # https://github.com/igfuw/libcloudphxx/blob/0b4e2455fba4f95c7387623fc21481a85e7b151f/src/impl/particles_impl_chem_strength.ipp#L50
    # https://en.wikipedia.org/wiki/Ionic_strength

    # H+ and OH-
    water = H + K_H2O / H

    # HSO4- and SO4 2-
    cz_S_VI = H * conc.S_VI / (H + K.HSO4) + 4 * K.HSO4 * conc.S_VI / (H + K.HSO4)

    # HCO3- and CO3 2-
    cz_CO2 = K.CO2 * H * conc.C_IV / (
        H * H + K.CO2 * H + K.CO2 * K.HCO3
    ) + 4 * K.CO2 * K.HCO3 * conc.C_IV / (H * H + K.CO2 * H + K.CO2 * K.HCO3)

    # HSO3- and HSO4 2-
    cz_SO2 = K.SO2 * H * conc.S_IV / (
        H * H + K.SO2 * H + K.SO2 * K.HSO3
    ) + 4 * K.SO2 * K.HSO3 * conc.S_IV / (H * H + K.SO2 * H + K.SO2 * K.HSO3)

    # NO3-
    cz_HNO3 = K.HNO3 * conc.N_V / (H + K.HNO3)

    # NH4+
    cz_NH3 = K.NH3 * H * conc.N_mIII / (K_H2O + K.NH3 * H)

    return 0.5 * (water + cz_S_VI + cz_CO2 + cz_SO2 + cz_HNO3 + cz_NH3)


@numba.njit(**{**conf.JIT_FLAGS, **{"parallel": False}})
def acidity_minfun(H, conc, K):
    ammonia = (conc.N_mIII * H * K.NH3) / (K_H2O + K.NH3 * H)
    nitric = conc.N_V * K.HNO3 / (H + K.HNO3)
    sulfous = (
        conc.S_IV * K.SO2 * (H + 2 * K.HSO3) / (H * H + H * K.SO2 + K.SO2 * K.HSO3)
    )
    water = K_H2O / H
    sulfuric = conc.S_VI * (H + 2 * K.HSO4) / (H + K.HSO4)
    carbonic = (
        conc.C_IV * K.CO2 * (H + 2 * K.HCO3) / (H * H + H * K.CO2 + K.CO2 * K.HCO3)
    )
    zero = H + ammonia - (nitric + sulfous + water + sulfuric + carbonic)
    return zero

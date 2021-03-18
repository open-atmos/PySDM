import numba
import numpy as np

from PySDM.backends.numba import conf
from PySDM.physics.constants import Md, R_str, Rd, M, K_H2O, ROOM_TEMP
from PySDM.physics.formulae import radius


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def dissolve_env_gases(super_droplet_ids, mole_amounts, env_mixing_ratio, henrysConstant, env_p, env_T,
                       env_rho_d, dt, dv, droplet_volume,
                       multiplicity, system_type, specific_gravity, alpha, diffusion_constant,
                       ksi):
    mole_amount_taken = 0
    for i in super_droplet_ids:  # TODO: idx?
        Mc = specific_gravity * Md
        Rc = R_str / Mc
        cinf = env_p / env_T / (Rd/env_mixing_ratio[0] + Rc) / Mc
        r_w = radius(volume=droplet_volume[i])
        v_avg = np.sqrt(8 * R_str * env_T / (np.pi * Mc))
        scale = (4 * r_w / (3 * v_avg * alpha) + r_w ** 2 / (3 * diffusion_constant))
        A_old = mole_amounts[i] / droplet_volume[i]
        A_new = (A_old + dt * cinf / scale) / (1 + dt / (scale * ksi[i] * henrysConstant * R_str * env_T))
        # TODO !!!!!!!!! A_new = (A_old + dt * ksi[i] * cinf / scale) / (1 + dt / (scale * ksi[i] * henrysConstant * R_str * env_T))
        new_mole_amount_per_real_droplet = A_new * droplet_volume[i]
        assert new_mole_amount_per_real_droplet >= 0

        mole_amount_taken += multiplicity[i] * (new_mole_amount_per_real_droplet - mole_amounts[i])
        mole_amounts[i] = new_mole_amount_per_real_droplet
    delta_mr = mole_amount_taken * specific_gravity * Md / (dv * env_rho_d)
    assert delta_mr <= env_mixing_ratio
    if system_type == 'closed':
        env_mixing_ratio -= delta_mr


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def oxidize(n_sd, cell_ids, do_chemistry_flag,
            k0, k1, k2, k3, K_SO2, K_HSO3,
            dt, droplet_volume,
            pH,
            O3,
            H2O2,
            S_IV,
            aqq_SO2,
            # output
            moles_O3,
            moles_H2O2,
            moles_S_IV,
            moles_S_VI
            ):
    # NB: magic_const in the paper is k4.
    # The value is fixed at 13 M^-1 (from Ania's Thesis)
    magic_const = 13 / M

    for i in range(n_sd):  # TODO: idx?
        if not do_chemistry_flag[i]:
            continue

        cid = cell_ids[i]
        H = pH2H(pH[i])
        SO2aq = S_IV[i] / aqq_SO2[i]

        # NB: This might not be entirely correct
        # https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JD092iD04p04171
        # https://www.atmos-chem-phys.net/16/1693/2016/acp-16-1693-2016.pdf

        # NB: There is also slight error due to "borrowing" compounds when
        # the concentration is close to 0. That way, if the rate is big enough,
        # it will consume more compound than there is.

        ozone = (k0[cid] + (k1[cid] * K_SO2[cid] / H) + (k2[cid] * K_SO2[cid] * K_HSO3[cid] / H**2)) * O3[i] * SO2aq
        peroxide = k3[cid] * K_SO2[cid] / (1 + magic_const * H) * H2O2[i] * SO2aq

        dconc_dt_O3 = -ozone
        dconc_dt_S_IV = -(ozone + peroxide)
        dconc_dt_H2O2 = -peroxide
        dconc_dt_S_VI = ozone + peroxide

        a = dt * droplet_volume[i]
        if (
            moles_O3[i] + dconc_dt_O3 * a < 0 or
            moles_S_IV[i] + dconc_dt_S_IV * a < 0 or
            moles_S_VI[i] + dconc_dt_S_VI * a < 0 or
            moles_H2O2[i] + dconc_dt_H2O2 * a < 0
        ):
            continue

        moles_O3[i] += dconc_dt_O3 * a
        moles_S_IV[i] += dconc_dt_S_IV * a
        moles_S_VI[i] += dconc_dt_S_VI * a
        moles_H2O2[i] += dconc_dt_H2O2 * a


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def calc_ionic_strength(H, N_mIII, N_V, C_IV, S_IV, S_VI, K_NH3, K_SO2, K_HSO3, K_HSO4, K_HCO3, K_CO2, K_HNO3):
    # Directly adapted
    # https://github.com/igfuw/libcloudphxx/blob/0b4e2455fba4f95c7387623fc21481a85e7b151f/src/impl/particles_impl_chem_strength.ipp#L50
    # https://en.wikipedia.org/wiki/Ionic_strength

    # H+ and OH-
    water = H + K_H2O / H

    # HSO4- and SO4 2-
    czS_VI = H * S_VI / (H + K_HSO4) + 4 * K_HSO4 * S_VI / (H + K_HSO4)

    # HCO3- and CO3 2-
    cz_CO2 = K_CO2 * H * C_IV / (H * H + K_CO2 * H + K_CO2 * K_HCO3) + \
        4 * K_CO2 * K_HCO3 * C_IV / (H * H + K_CO2 * H + K_CO2 * K_HCO3)

    # HSO3- and HSO4 2-
    cz_SO2 = K_SO2 * H * S_IV / (H * H + K_SO2 * H + K_SO2 * K_HSO3) + \
        4 * K_SO2 * K_HSO3 * S_IV / (H * H + K_SO2 * H + K_SO2 * K_HSO3)

    # NO3-
    cz_HNO3 = K_HNO3 * N_V / (H + K_HNO3)

    # NH4+
    cz_NH3 = K_NH3 * H * N_mIII / (K_H2O + K_NH3 * H)

    return 0.5 * (water + czS_VI + cz_CO2 + cz_SO2 + cz_HNO3 + cz_NH3)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def concentration(H, N_mIII, N_V, C_IV, S_IV, S_VI, K_NH3, K_SO2, K_HSO3, K_HSO4, K_HCO3, K_CO2, K_HNO3):
    ammonia = (N_mIII * H * K_NH3) / (K_H2O + K_NH3 * H)
    nitric = N_V * K_HNO3 / (H + K_HNO3)
    sulfous = S_IV * K_SO2 * (H + 2 * K_HSO3) / (H * H + H * K_SO2 + K_SO2 * K_HSO3)
    water = K_H2O / H
    sulfuric = S_VI * (H + 2 * K_HSO4) / (H + K_HSO4)
    carbonic = C_IV * K_CO2 * (H + 2 * K_HCO3) / (H * H + H * K_CO2 + K_CO2 * K_HCO3)
    zero = H + ammonia - (nitric + sulfous + water + sulfuric + carbonic)
    return zero


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def pH2H(pH):
    return 10**(-pH) * 1e3


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def H2pH(H):
    return -np.log10(H * 1e-3)


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def vant_hoff(K, dH, T, *, T_0=ROOM_TEMP):
    return K * np.exp(-dH / R_str * (1 / T - 1/T_0))


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def tdep2enthalpy(tdep):
    return -tdep * R_str


@numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False}})
def arrhenius(A, Ea, T=ROOM_TEMP):
    return A * np.exp(-Ea / (R_str * T))

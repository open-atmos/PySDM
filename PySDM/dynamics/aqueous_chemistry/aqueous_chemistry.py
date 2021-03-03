import numba
import numpy as np
from PySDM.backends.numba.numba_helpers import temperature_pressure_RH
from .support import EqConst
from PySDM.physics.constants import H_u, dT_u, _weight, Md, M, si, R_str, Rd
from PySDM.physics.formulae import mole_fraction_2_mixing_ratio, mixing_ratio_2_partial_pressure, radius
from scipy import optimize


HENRY_CONST = {
    "HNO3": EqConst(2.1e5 * H_u, 0 * dT_u),
    "H2O2": EqConst(7.45e4 * H_u, 7300 * dT_u),
    "NH3":  EqConst(62 * H_u, 4110 * dT_u),
    "SO2":  EqConst(1.23 * H_u, 3150 * dT_u),
    "CO2":  EqConst(3.4e-2 * H_u, 2440 * dT_u),
    "O3":   EqConst(1.13e-2 * H_u, 2540 * dT_u),
}

EQUILIBRIUM_CONST = {  # Reaction Specific units, K
    # ("HNO3(aq) = H+ + NO3-", 15.4, 0),
    "K_HNO3": EqConst(15.4 * M, 0 * dT_u),
    # ("H2SO3(aq) = H+ + HSO3-", 1.54*10**-2 * KU, 1960),
    "K_SO2":  EqConst(1.3e-2 * M, 1960 * dT_u),
    # ("NH4+ = NH3(aq) + H+", 10**-9.25 * M, 0),
    "K_NH3":  EqConst(1.7e-5 * M, -450 * dT_u),
    # ("H2CO3(aq) = H+ + HCO3-", 4.3*10**-7 * KU, -1000),
    "K_CO2":  EqConst(4.3e-7 * M, -1000 * dT_u),
    # ("HSO3- = H+ + SO3-2", 6.6*10**-8 * KU, 1500),
    "K_HSO3": EqConst(6.6e-8 * M, 1500 * dT_u),
    # ("HCO3- = H+ + CO3-2", 4.68*10**-11 * KU, -1760),
    "K_HCO3": EqConst(4.68e-11 * M, -1760 * dT_u),
    # ("HSO4- = H+ + SO4-2", 1.2*10**-2 * KU, 2720),
    "K_HSO4": EqConst(1.2e-2 * M, 2720 * dT_u)
}

K_H2O = 1e-14 * M * M

DIFFUSION_CONST = {
    "HNO3": 65.25e-6 * si.m**2 / si.s,
    "H2O2": 87.00e-6 * si.m**2 / si.s,
    "NH3":  19.78e-6 * si.m**2 / si.s,
    "SO2":  10.89e-6 * si.m**2 / si.s,
    "CO2":  13.81e-6 * si.m**2 / si.s,
    "O3":   14.44e-6 * si.m**2 / si.s,
}


MASS_ACCOMMODATION_COEFFICIENTS = {
    "HNO3": 0.05,
    "H2O2": 0.018,
    "NH3":  0.05,
    "SO2":  0.035,
    "CO2":  0.05,
    "O3":   0.00053
}

AQUEOUS_COMPOUNDS = {
    "S_IV": ("SO2 H2O", "HSO3", "SO3"),
    "O3": ("O3",),
    "H2O2": ("H2O2",),
    "C_IV": ("CO2 H2O", "HCO3", "CO3"),
    "N_V": ("HNO3", "NO3"),
    "N_mIII": ("NH4", "H2O NH3"),
    "S_VI": ("SO4", "HSO4"),
    "H": ("H",)
}

GASEOUS_COMPOUNDS = {
    "N_V": "HNO3",
    "H2O2": "H2O2",
    "N_mIII": "NH3",
    "S_IV": "SO2",
    "C_IV": "CO2",
    "O3": "O3"
}

def aqq_CO2(H):
    return (1 + EQUILIBRIUM_CONST["K_CO2"] * (1 / H + EQUILIBRIUM_CONST["K_HCO3"] / (H ** 2)))

def aqq_SO2(H):
    return (1 + EQUILIBRIUM_CONST["K_SO2"] * (1 / H + EQUILIBRIUM_CONST["K_HSO3"] / (H ** 2)))

def aqq_NH3(H):
    return (1 + EQUILIBRIUM_CONST["K_NH3"] / K_H2O * H)

def aqq_HNO3(H):
    return (1 + EQUILIBRIUM_CONST["K_HNO3"] / H)

def aqq(_):
    return 1

AQQ = {
    "CO2": aqq_CO2,
    "SO2": aqq_SO2,
    "NH3": aqq_NH3,
    "HNO3": aqq_HNO3,
    "O3": aqq,
    "H2O2": aqq
}

SPECIFIC_GRAVITY = {
    compound: _weight(compound) / Md for compound in {*GASEOUS_COMPOUNDS.values()}
}
for compounds in AQUEOUS_COMPOUNDS.values():
    for compound in compounds:
        SPECIFIC_GRAVITY[compound] = _weight(compound) / Md


def dissolve_env_gases(super_droplet_ids, mole_amounts, env_mixing_ratio, henrysConstant, env_p, env_T,
                       env_rho_d, dt, dv, droplet_volume,
                       multiplicity, system_type, specific_gravity, alpha, diffusion_constant,
                       hconcdep, H_conc):
    mole_amount_taken = 0
    for i in super_droplet_ids:
        Mc = specific_gravity * Md
        Rc = R_str / Mc
        cinf = env_p / env_T / (Rd/env_mixing_ratio + Rc) / Mc
        Heff = henrysConstant * hconcdep(H_conc[i])
        r_w = radius(volume=droplet_volume[i])
        v_avg = np.sqrt(8 * R_str * env_T / (np.pi * Mc))
        scale = (4 * r_w / (3 * v_avg * alpha) + r_w ** 2 / (3 * diffusion_constant))
        concentration = (mole_amounts.data[i]/droplet_volume[i] + dt * cinf / scale) / (1 + dt / (scale * Heff * R_str * env_T))

        new_mole_amount_per_real_droplet = concentration * droplet_volume[i]
        mole_amount_taken += multiplicity[i] * (new_mole_amount_per_real_droplet - mole_amounts[i])
        mole_amounts.data[i] = new_mole_amount_per_real_droplet
        assert mole_amounts[i] >= 0
    delta_mr = mole_amount_taken * specific_gravity * Md / (dv * env_rho_d)
    assert delta_mr <= env_mixing_ratio
    if system_type == 'closed':
        env_mixing_ratio -= delta_mr


def equilibrate_pH(super_droplet_ids, particles, env_T):
    N_III = particles["conc_N_mIII"].data
    N_V = particles["conc_N_V"].data
    C_IV = particles["conc_C_IV"].data
    S_IV = particles["conc_S_IV"].data
    S_VI = particles["conc_S_VI"].data
    volume = particles["volume"].data
    moles_H = particles["moles_H"].data

    K_NH3 = EQUILIBRIUM_CONST["K_NH3"].at(env_T)
    K_SO2 = EQUILIBRIUM_CONST["K_SO2"].at(env_T)
    K_HSO3 = EQUILIBRIUM_CONST["K_HSO3"].at(env_T)
    K_HSO4 = EQUILIBRIUM_CONST["K_HSO4"].at(env_T)
    K_HCO3 = EQUILIBRIUM_CONST["K_HCO3"].at(env_T)
    K_CO2 = EQUILIBRIUM_CONST["K_CO2"].at(env_T)
    K_HNO3 = EQUILIBRIUM_CONST["K_HNO3"].at(env_T)

    def concentration(H_i, args):
        i = args
        ammonia = (N_III[i] * H_i * K_NH3) / (K_H2O + K_NH3 * H_i)
        nitric = N_V[i] * K_HNO3 / (H_i + K_HNO3)
        sulfous = S_IV[i] * K_SO2 * (H_i + 2 * K_HSO3) / (H_i * H_i + H_i * K_SO2 + K_SO2 * K_HSO3)
        water = K_H2O / H_i
        sulfuric = S_VI[i] * (H_i + 2 * K_HSO4) / (H_i + K_HSO4)
        carbonic = C_IV[i] * K_CO2 * (H_i + 2 * K_HCO3) / (H_i * H_i + H_i * K_CO2 + K_CO2 * K_HCO3)
        zero = H_i + ammonia - (nitric + sulfous + water + sulfuric + carbonic)
        return zero

    pH_min=-1
    pH_max=9
    H_min = 10**(-pH_max) * (si.m**3 / si.litre)
    H_max = 10**(-pH_min) * (si.m**3 / si.litre)
    for i in super_droplet_ids:
        args = (i,)
        result = optimize.root_scalar(concentration, x0=H_min, x1=H_max, args=args)
        assert result.converged
        moles_H[i] = result.root * volume[i]


def calc_ionic_strength(*, Hp, N_III, N_V, C_IV, S_IV, S_VI , env_T):

    K_NH3 = EQUILIBRIUM_CONST["K_NH3"].at(env_T)
    K_SO2 = EQUILIBRIUM_CONST["K_SO2"].at(env_T)
    K_HSO3 = EQUILIBRIUM_CONST["K_HSO3"].at(env_T)
    K_HSO4 = EQUILIBRIUM_CONST["K_HSO4"].at(env_T)
    K_HCO3 = EQUILIBRIUM_CONST["K_HCO3"].at(env_T)
    K_CO2 = EQUILIBRIUM_CONST["K_CO2"].at(env_T)
    K_HNO3 = EQUILIBRIUM_CONST["K_HNO3"].at(env_T)

    # Directly adapted
    # https://github.com/igfuw/libcloudphxx/blob/0b4e2455fba4f95c7387623fc21481a85e7b151f/src/impl/particles_impl_chem_strength.ipp#L50
    # https://en.wikipedia.org/wiki/Ionic_strength

    # H+ and OH-
    water = Hp + K_H2O / Hp

    # HSO4- and SO4 2-
    czS_VI = Hp * S_VI / (Hp + K_HSO4) + 4 * K_HSO4 * S_VI / (Hp + K_HSO4)

    # HCO3- and CO3 2-
    cz_CO2 = K_CO2 * Hp * C_IV / (Hp * Hp + K_CO2 * Hp + K_CO2 * K_HCO3) + \
        4 * K_CO2 * K_HCO3 * C_IV / (Hp * Hp + K_CO2 * Hp + K_CO2 * K_HCO3)

    # HSO3- and HSO4 2-
    cz_SO2 = K_SO2 * Hp * S_IV / (Hp * Hp + K_SO2 * Hp + K_SO2 * K_HSO3) + \
        4 * K_SO2 * K_HSO3 * S_IV / (Hp * Hp + K_SO2 * Hp + K_SO2 * K_HSO3)

    # NO3-
    cz_HNO3 = K_HNO3 * N_V / (Hp + K_HNO3)

    # NH4+
    cz_NH3 = K_NH3 * Hp * N_III / (K_H2O + K_NH3 * Hp)

    return 0.5 * (water + czS_VI + cz_CO2 + cz_SO2 + cz_HNO3 + cz_NH3)


def oxidation_factory(*, k0, k1, k2, k3, K_SO2, K_HSO3, magic_const=13 / M, **kwargs):
    # NB: magic_const in the paper is k4.
    # The value is fixed at 13 M^-1 (from dr Jaruga's Thesis)

    # NB: This might not be entirely correct
    # https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JD092iD04p04171
    # https://www.atmos-chem-phys.net/16/1693/2016/acp-16-1693-2016.pdf

    # NB: There is also slight error due to "borrowing" compounds when
    # the concentration is close to 0. That way, if the rate is big enough,
    # it will consume more compound than there is.

    def oxidation(x):
        x[x < 0] = 0
        H, O3, SO2aq, H2O2, HSO4m = x
        ozone = ((k0) + (k1 * K_SO2 / H) + (k2 * K_SO2 * K_HSO3 / (H ** 2))) * O3 * SO2aq
        peroxide = k3 * K_SO2 / (1 + magic_const * H) * H2O2 * SO2aq
        return (0, -ozone, -(ozone + peroxide), -peroxide, ozone + peroxide)
    return oxidation


class AqueousChemistry:
    def __init__(self, environment_mole_fractions, system_type):
        self.environment_mixing_ratios = {}
        for key, compound in GASEOUS_COMPOUNDS.items():
            shape = (1,)  # TODO #157
            self.environment_mixing_ratios[compound] = np.full(
                shape,
                mole_fraction_2_mixing_ratio(environment_mole_fractions[compound], SPECIFIC_GRAVITY[compound])
            )
        self.mesh = None
        self.core = None
        self.env = None
        assert system_type in ('open', 'closed')
        self.system_type = system_type

    def register(self, builder):
        self.mesh = builder.core.mesh
        self.core = builder.core
        self.env = builder.core.env
        for key in AQUEOUS_COMPOUNDS.keys():
            builder.request_attribute("conc_" + key)

    def __call__(self):
        n_cell = self.mesh.n_cell
        n_threads = 1  # TODO #157
        cell_order = np.arange(n_cell)  # TODO #157
        cell_start_arg = self.core.particles.cell_start.data
        idx = self.core.particles._Particles__idx

        rhod = self.env["rhod"]
        thd = self.env["thd"]
        qv = self.env["qv"]
        prhod = self.env.get_predicted("rhod")

        # TODO #157: same code in condensation
        n_substep = 2
        for _ in range(n_substep):
            for thread_id in numba.prange(n_threads):
                for i in range(thread_id, n_cell, n_threads):
                    cell_id = cell_order[i]

                    cell_start = cell_start_arg[cell_id]
                    cell_end = cell_start_arg[cell_id + 1]
                    n_sd_in_cell = cell_end - cell_start
                    if n_sd_in_cell == 0:
                        continue

                    rhod_mean = (prhod[cell_id] + rhod[cell_id]) / 2
                    T, p, RH = temperature_pressure_RH(rhod_mean, thd[cell_id], qv[cell_id])  # TODO #157: this is surely already computed elsewhere!

                    super_droplet_ids = []
                    for sd_id in idx[cell_start:cell_end]:
                        if calc_ionic_strength(
                            Hp=self.core.particles['conc_H'][sd_id],
                            N_III=self.core.particles['conc_N_mIII'][sd_id],
                            N_V=self.core.particles['conc_N_V'][sd_id],
                            C_IV=self.core.particles['conc_C_IV'][sd_id],
                            S_IV=self.core.particles['conc_S_IV'][sd_id],
                            S_VI=self.core.particles['conc_S_VI'][sd_id],
                            env_T=T
                        ) < 0.02 * M:
                            super_droplet_ids.append(sd_id)

                    for key, compound in GASEOUS_COMPOUNDS.items():
                        dissolve_env_gases(
                            super_droplet_ids=super_droplet_ids,
                            mole_amounts=self.core.particles['moles_'+key],
                            env_mixing_ratio=self.environment_mixing_ratios[compound][cell_id:cell_id+1],
                            henrysConstant=HENRY_CONST[compound].at(T),  # mol m−3 Pa−1
                            env_p=p,
                            env_T=T,
                            env_rho_d=rhod_mean,
                            dt=self.core.dt/n_substep,
                            dv=self.mesh.dv,
                            droplet_volume=self.core.particles["volume"],
                            multiplicity=self.core.particles["n"],
                            system_type=self.system_type,
                            specific_gravity=SPECIFIC_GRAVITY[compound],
                            alpha=MASS_ACCOMMODATION_COEFFICIENTS[compound],
                            diffusion_constant=DIFFUSION_CONST[compound],
                            hconcdep=AQQ[compound],
                            H_conc=self.core.particles["conc_H"]
                        )
                        self.core.particles.attributes[f'moles_{key}'].mark_updated()

                    equilibrate_pH(
                        super_droplet_ids=super_droplet_ids,
                        particles=self.core.particles,
                        env_T=T
                    )
                    self.core.particles.attributes['moles_H'].mark_updated()


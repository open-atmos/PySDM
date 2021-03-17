from PySDM.attributes.impl.intensive_attribute import DerivedAttribute
from PySDM.physics.constants import convert_to
from PySDM.physics import si
from PySDM.dynamics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS, EQUILIBRIUM_CONST, K_H2O, M, H2pH, FLAG
from PySDM.backends.numba.numba_helpers import bisec
import numba
from PySDM.backends.numba.conf import JIT_FLAGS

# TODO (iterate in logarithm?)
pH_min=-1
pH_max=14

H_min = 10 ** (-pH_max) * (si.m ** 3 / si.litre)
H_max = 10 ** (-pH_min) * (si.m ** 3 / si.litre)


@numba.njit(**JIT_FLAGS)
def concentration(H, N_mIII, N_V, C_IV, S_IV, S_VI, K_NH3, K_SO2, K_HSO3, K_HSO4, K_HCO3, K_CO2, K_HNO3):
    ammonia = (N_mIII * H * K_NH3) / (K_H2O + K_NH3 * H)
    nitric = N_V * K_HNO3 / (H + K_HNO3)
    sulfous = S_IV * K_SO2 * (H + 2 * K_HSO3) / (H * H + H * K_SO2 + K_SO2 * K_HSO3)
    water = K_H2O / H
    sulfuric = S_VI * (H + 2 * K_HSO4) / (H + K_HSO4)
    carbonic = C_IV * K_CO2 * (H + 2 * K_HCO3) / (H * H + H * K_CO2 + K_CO2 * K_HCO3)
    zero = H + ammonia - (nitric + sulfous + water + sulfuric + carbonic)
    return zero


def equilibrate_H(env_T, N_mIII, N_V, C_IV, S_IV, S_VI):
    K_NH3 = EQUILIBRIUM_CONST["K_NH3"].at(env_T)
    K_SO2 = EQUILIBRIUM_CONST["K_SO2"].at(env_T)
    K_HSO3 = EQUILIBRIUM_CONST["K_HSO3"].at(env_T)
    K_HSO4 = EQUILIBRIUM_CONST["K_HSO4"].at(env_T)
    K_HCO3 = EQUILIBRIUM_CONST["K_HCO3"].at(env_T)
    K_CO2 = EQUILIBRIUM_CONST["K_CO2"].at(env_T)
    K_HNO3 = EQUILIBRIUM_CONST["K_HNO3"].at(env_T)

    args = (
        N_mIII, N_V, C_IV, S_IV, S_VI,
        K_NH3, K_SO2, K_HSO3, K_HSO4, K_HCO3, K_CO2, K_HNO3
    )
    H = bisec(concentration, H_min, H_max-H_min, args, rtol=1e-6)  # TODO: pass as arg

    if calc_ionic_strength(H, *args) > 0.02 * M:
        return FLAG
    else:
        return H2pH(H)


@numba.njit(**JIT_FLAGS)
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


class pH(DerivedAttribute):
    def __init__(self, builder):
        self.conc = {}
        for k, v in AQUEOUS_COMPOUNDS.items():
            if len(v) > 1:
                self.conc[k] = builder.get_attribute('conc_' + k)
        super().__init__(builder, name='pH', dependencies=self.conc.values())
        self.environment = builder.core.environment
        self.cell_id = builder.get_attribute('cell id')
        self.particles = builder.core

    def recalculate(self):
        T = self.environment['T'].data
        cell_id = self.cell_id.get().data

        N_mIII = self.conc["N_mIII"].get().data
        N_V = self.conc["N_V"].get().data
        C_IV = self.conc["C_IV"].get().data
        S_IV = self.conc["S_IV"].get().data
        S_VI = self.conc["S_VI"].get().data

        # TODO #435
        for i in range(len(self.data)):  # TODO #347 move to backend and parallelize
            self.data.data[i] = equilibrate_H(T[cell_id[i]], N_mIII[i], N_V[i], C_IV[i], S_IV[i], S_VI[i])



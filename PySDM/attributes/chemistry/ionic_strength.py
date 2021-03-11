from ..impl.derived_attribute import DerivedAttribute
from ...dynamics.aqueous_chemistry.support import EQUILIBRIUM_CONST, K_H2O


class IonicStrength(DerivedAttribute):
    pass


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
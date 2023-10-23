import numpy as np

from PySDM.dynamics.isotopic_fractionation import HEAVY_ISOTOPES


def equilibrate_isotopologues(
    *, water_mass: np.ndarray, environment, cell_id: np.ndarray = None
):
    """
    returns a dictionary of attribute_name:np.ndarray pairs containing mole_concentrations
    of heavy isotopes of H & O in equilibrium with ambient temperature
    """
    if cell_id is None:
        cell_id = np.zeros_like(water_mass, dtype=int)

    const = environment.particulator.formulae.constants
    ambient_delta_vapour = {
        isotope: environment[f"delta_{isotope}"].to_ndarray()
        for isotope in HEAVY_ISOTOPES
    }
    ambient_temperature = environment[f"T"].to_ndarray()

    if (ambient_temperature < const.T0).any():
        raise ValueError("only liquid water equilibration supported (no ice)")

    # TODO

    return {f"moles_{isotope}": np.zeros_like(water_mass) for isotope in HEAVY_ISOTOPES}

import numpy as np
from PySDM.backends.numba.impl._chemistry_methods import ChemistryMethods
from PySDM.physics import si, Formulae
from PySDM.physics.aqueous_chemistry.support import KineticConsts, EquilibriumConsts, \
    DISSOCIATION_FACTORS
from PySDM.physics.constants import T_STP, pi_4_3
import pytest


formulae = Formulae()


class SUT(ChemistryMethods):
    def __init__(self):
        self.formulae = formulae
        super().__init__()


kinetic_consts = KineticConsts(formulae)
equilibrium_consts = EquilibriumConsts(formulae)

T = T_STP

k0 = np.full(1, kinetic_consts.KINETIC_CONST['k0'].at(T))
k1 = np.full(1, kinetic_consts.KINETIC_CONST['k1'].at(T))
k2 = np.full(1, kinetic_consts.KINETIC_CONST['k2'].at(T))
k3 = np.full(1, kinetic_consts.KINETIC_CONST['k3'].at(T))
k4 = kinetic_consts.KINETIC_CONST['k4']
K_SO2 = np.full(1, equilibrium_consts.EQUILIBRIUM_CONST['K_SO2'].at(T))
K_HSO3 = np.full(1, equilibrium_consts.EQUILIBRIUM_CONST['K_HSO3'].at(T))

volume = pi_4_3 * (1 * si.um)**3
pH = 5.
n_sd = 1
eqc = {
    k: np.full(n_sd, equilibrium_consts.EQUILIBRIUM_CONST[k].at(T))
    for k in ('K_HSO3', 'K_SO2')
}
cell_ids = np.zeros(n_sd, dtype=int)
H = formulae.trivia.pH2H(pH)
DF = DISSOCIATION_FACTORS['SO2'](H, eqc, 0)


@pytest.mark.parametrize('conc', [
    {
        'input': {},
        'output': {'S_IV': 0., 'S_VI': 0., 'H2O2': 0., 'O3': 0.}
    },
    # {
    #     'input': {'S_IV': 44., 'O3': 22.},
    #     'output': {'S_VI': (k0 + k1 * K_SO2 / H + k2 * K_SO2 * K_HSO3 / H**2) * 22 * DF},
    # }
])
def test_oxidation(conc):
    # Arrange
    sut = SUT()
    dt = 1

    moles = {
        k: np.full(n_sd, 0. if k not in conc['input'] else conc['input'][k] * volume)
        for k in ('S_IV', 'S_VI', 'H2O2', 'O3')
    }

    # Act
    sut.oxidation(
        n_sd=n_sd,
        cell_ids=cell_ids,
        do_chemistry_flag=np.full(n_sd, True),
        k0=k0,
        k1=k1,
        k2=k2,
        k3=k3,
        k4=k4,
        K_SO2=K_SO2,
        K_HSO3=K_HSO3,
        dt=dt,
        droplet_volume=np.full(n_sd, volume),
        pH=np.full(n_sd, pH),
        dissociation_factor_SO2=np.full(n_sd, DF),
        # input/output
        moles_O3=moles['O3'],
        moles_H2O2=moles['H2O2'],
        moles_S_IV=moles['S_IV'],
        moles_S_VI=moles['S_VI']
    )

    # Assert
    for k in conc['output'].keys():
        assert moles[k] / volume == conc['output'][k]

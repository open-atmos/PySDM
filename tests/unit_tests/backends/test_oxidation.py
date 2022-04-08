# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PySDM import Formulae
from PySDM.backends.impl_numba.methods.chemistry_methods import ChemistryMethods
from PySDM.backends.impl_numba.storage import Storage
from PySDM.dynamics.impl.chemistry_utils import (
    DISSOCIATION_FACTORS,
    EquilibriumConsts,
    KineticConsts,
    k4,
)
from PySDM.physics import si
from PySDM.physics.constants import PI_4_3
from PySDM.physics.constants_defaults import T_STP

formulae = Formulae()


class SUT(ChemistryMethods):
    def __init__(self):
        self.formulae = formulae
        super().__init__()


kinetic_consts = KineticConsts(formulae)
equilibrium_consts = EquilibriumConsts(formulae)

k0 = Storage.from_ndarray(np.full(1, kinetic_consts.KINETIC_CONST["k0"].at(T_STP)))
k1 = Storage.from_ndarray(np.full(1, kinetic_consts.KINETIC_CONST["k1"].at(T_STP)))
k2 = Storage.from_ndarray(np.full(1, kinetic_consts.KINETIC_CONST["k2"].at(T_STP)))
k3 = Storage.from_ndarray(np.full(1, kinetic_consts.KINETIC_CONST["k3"].at(T_STP)))
K_SO2 = Storage.from_ndarray(
    np.full(1, equilibrium_consts.EQUILIBRIUM_CONST["K_SO2"].at(T_STP))
)
K_HSO3 = Storage.from_ndarray(
    np.full(1, equilibrium_consts.EQUILIBRIUM_CONST["K_HSO3"].at(T_STP))
)

volume = PI_4_3 * (1 * si.um) ** 3
pH = 5.0
n_sd = 1
eqc = {
    k: Storage.from_ndarray(
        np.full(n_sd, equilibrium_consts.EQUILIBRIUM_CONST[k].at(T_STP))
    )
    for k in ("K_HSO3", "K_SO2")
}
cell_ids = Storage.from_ndarray(np.zeros(n_sd, dtype=int))
H = formulae.trivia.pH2H(pH)
DF = DISSOCIATION_FACTORS["SO2"](H, eqc, 0)

O3_react_consts = (
    k0.data[0]
    + k1.data[0] * K_SO2.data[0] / H
    + k2.data[0] * K_SO2.data[0] * K_HSO3.data[0] / H**2
)
H2O2_react_consts = k3.data[0] * K_SO2.data[0] / (1.0 + k4 * H)

O3_init = 1e-4
H2O2_init = 1e-4
S_IV_init = 1e-4
S_VI_init = 1e-4


@pytest.mark.parametrize(
    "conc",
    [
        pytest.param(
            {
                "input": {"S_IV": 0.0, "S_VI": 0.0, "H2O2": 0.0, "O3": 0.0},
                "output": {"S_IV": 0.0, "S_VI": 0.0, "H2O2": 0.0, "O3": 0.0},
            },
            id="zeros",
        ),
        pytest.param(
            {
                "input": {"S_IV": S_IV_init, "O3": O3_init, "H2O2": 0.0, "S_VI": 0.0},
                "output": {
                    "S_VI": O3_react_consts * O3_init * S_IV_init / DF,
                    "O3": -O3_react_consts * O3_init * S_IV_init / DF,
                    "H2O2": 0.0,
                    "S_IV": -O3_react_consts * O3_init * S_IV_init / DF,
                },
            },
            id="ozone",
        ),
        pytest.param(
            {
                "input": {"S_IV": S_IV_init, "O3": 0, "H2O2": H2O2_init, "S_VI": 0.0},
                "output": {
                    "S_VI": H2O2_react_consts * H2O2_init * S_IV_init / DF,
                    "O3": 0.0,
                    "H2O2": -H2O2_react_consts * H2O2_init * S_IV_init / DF,
                    "S_IV": -H2O2_react_consts * H2O2_init * S_IV_init / DF,
                },
            },
            id="hydrogen peroxide",
        ),
        pytest.param(
            {
                "input": {
                    "S_IV": S_IV_init,
                    "O3": O3_init,
                    "H2O2": H2O2_init,
                    "S_VI": 0.0,
                },
                "output": {
                    "S_VI": (H2O2_react_consts * H2O2_init + O3_react_consts * O3_init)
                    * S_IV_init
                    / DF,
                    "O3": -O3_react_consts * O3_init * S_IV_init / DF,
                    "H2O2": -H2O2_react_consts * H2O2_init * S_IV_init / DF,
                    "S_IV": -(H2O2_react_consts * H2O2_init + O3_react_consts * O3_init)
                    * S_IV_init
                    / DF,
                },
            },
            id="all with no initial S_VI",
        ),
        pytest.param(
            {
                "input": {
                    "S_IV": S_IV_init,
                    "O3": O3_init,
                    "H2O2": H2O2_init,
                    "S_VI": S_VI_init,
                },
                "output": {
                    "S_VI": (H2O2_react_consts * H2O2_init + O3_react_consts * O3_init)
                    * S_IV_init
                    / DF,
                    "O3": -O3_react_consts * O3_init * S_IV_init / DF,
                    "H2O2": -H2O2_react_consts * H2O2_init * S_IV_init / DF,
                    "S_IV": -(H2O2_react_consts * H2O2_init + O3_react_consts * O3_init)
                    * S_IV_init
                    / DF,
                },
            },
            id="all",
        ),
    ],
)
@pytest.mark.parametrize("dt", (1, 0.1))
def test_oxidation(conc, dt):
    # Arrange
    sut = SUT()

    moles = {
        k: Storage.from_ndarray(
            np.full(n_sd, 0.0 if k not in conc["input"] else conc["input"][k] * volume)
        )
        for k in ("S_IV", "S_VI", "H2O2", "O3")
    }

    # Act
    sut.oxidation(
        n_sd=n_sd,
        cell_ids=cell_ids,
        do_chemistry_flag=Storage.from_ndarray(np.full(n_sd, True)),
        k0=k0,
        k1=k1,
        k2=k2,
        k3=k3,
        K_SO2=K_SO2,
        K_HSO3=K_HSO3,
        timestep=dt,
        droplet_volume=Storage.from_ndarray(np.full(n_sd, volume)),
        pH=Storage.from_ndarray(np.full(n_sd, pH)),
        dissociation_factor_SO2=Storage.from_ndarray(np.full(n_sd, DF)),
        # input/output
        moles_O3=moles["O3"],
        moles_H2O2=moles["H2O2"],
        moles_S_IV=moles["S_IV"],
        moles_S_VI=moles["S_VI"],
    )

    # Assert
    for k in conc["output"].keys():
        np.testing.assert_allclose(
            actual=moles[k].data / volume - conc["input"][k],
            desired=conc["output"][k] * dt,
            rtol=1e-11,
        )

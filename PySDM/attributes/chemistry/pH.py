from PySDM.attributes.impl.intensive_attribute import DerivedAttribute
from PySDM.backends.numba.impl._chemistry_methods import calc_ionic_strength, concentration, H2pH, pH2H
from PySDM.dynamics.aqueous_chemistry.support import AQUEOUS_COMPOUNDS, M
from PySDM.backends.numba.numba_helpers import bisec

# TODO #439 (iterate in logarithm?)
pH_min = -1
pH_max = 14
H_min = pH2H(pH_max)
H_max = pH2H(pH_min)


def equilibrate_H(equilibrium_consts, cell_id, N_mIII, N_V, C_IV, S_IV, S_VI):
    K_NH3 = equilibrium_consts["K_NH3"].data[cell_id]
    K_SO2 = equilibrium_consts["K_SO2"].data[cell_id]
    K_HSO3 = equilibrium_consts["K_HSO3"].data[cell_id]
    K_HSO4 = equilibrium_consts["K_HSO4"].data[cell_id]
    K_HCO3 = equilibrium_consts["K_HCO3"].data[cell_id]
    K_CO2 = equilibrium_consts["K_CO2"].data[cell_id]
    K_HNO3 = equilibrium_consts["K_HNO3"].data[cell_id]

    args = (
        N_mIII, N_V, C_IV, S_IV, S_VI,
        K_NH3, K_SO2, K_HSO3, K_HSO4, K_HCO3, K_CO2, K_HNO3
    )
    H = bisec(concentration, H_min, H_max - H_min, args, rtol=1e-6)  # TODO #439: pass rtol as arg
    flag = calc_ionic_strength(H, *args) <= 0.02 * M
    return H2pH(H), flag


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
        dynamic = self.particles.dynamics['AqueousChemistry']
        equilibrium_consts = dynamic.equilibrium_consts
        do_chemistry_flag = dynamic.do_chemistry_flag

        cell_id = self.cell_id.get().data

        N_mIII = self.conc["N_mIII"].get().data
        N_V = self.conc["N_V"].get().data
        C_IV = self.conc["C_IV"].get().data
        S_IV = self.conc["S_IV"].get().data
        S_VI = self.conc["S_VI"].get().data

        # TODO #435
        for i in range(len(self.data)):  # TODO #347 move to backend and parallelize

            pH, flag = equilibrate_H(equilibrium_consts, cell_id[i], N_mIII[i], N_V[i], C_IV[i], S_IV[i], S_VI[i])
            do_chemistry_flag.data[i] = flag
            self.data.data[i] = pH



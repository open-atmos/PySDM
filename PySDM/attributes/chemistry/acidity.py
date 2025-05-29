"""
pH calculated by finding equilibrium hydrogen ion concentration
"""

from PySDM.attributes.impl import DerivedAttribute, register_attribute
from PySDM.backends.impl_numba.methods.chemistry_methods import _conc
from PySDM.dynamics.impl.chemistry_utils import AQUEOUS_COMPOUNDS


@register_attribute(name="pH")
class Acidity(DerivedAttribute):
    def __init__(self, builder):
        self.conc = {}
        for key, val in AQUEOUS_COMPOUNDS.items():
            if len(val) > 1:
                self.conc[key] = builder.get_attribute("conc_" + key)
        super().__init__(builder, name="pH", dependencies=self.conc.values())
        self.environment = builder.particulator.environment
        self.cell_id = builder.get_attribute("cell id")

    def allocate(self, idx):
        super().allocate(idx)
        self.data.fill(self.formulae.constants.pH_w)

    def recalculate(self):
        dynamic = self.particulator.dynamics["AqueousChemistry"]

        self.particulator.backend.equilibrate_H(
            equilibrium_consts=dynamic.equilibrium_consts,
            cell_id=self.cell_id.get(),
            conc=_conc(
                N_mIII=self.conc["N_mIII"].get(),
                N_V=self.conc["N_V"].get(),
                C_IV=self.conc["C_IV"].get(),
                S_IV=self.conc["S_IV"].get(),
                S_VI=self.conc["S_VI"].get(),
            ),
            do_chemistry_flag=dynamic.do_chemistry_flag,
            pH=self.data,
            H_min=dynamic.pH_H_min,
            H_max=dynamic.pH_H_max,
            ionic_strength_threshold=dynamic.ionic_strength_threshold,
            rtol=dynamic.pH_rtol,
        )

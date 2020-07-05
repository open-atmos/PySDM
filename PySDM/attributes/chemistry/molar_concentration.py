"""
Created at 01.06.2020

@author: Grzegorz Łazarski
"""

from PySDM.attributes.tensive_attribute import TensiveAttribute

# TODO Duplicated info, needs to be kept in sync with
# chemical_reaction.oxidation.dynamic.COMPOUNDS

COMPOUNDS = [
    "SO2",
    "O3",
    "H2O2",
    "CO2",
    "HNO3",
    "NH3",
    "HSO4m",
    "Hp"]


class ConcentrationImpl(TensiveAttribute):

    def __init__(self, particles_builder, *, name):
        super().__init__(particles_builder, name=name, extensive=False)


def Concentration(what):
    def _constructor(pb):
        return ConcentrationImpl(pb, name=what)
    return _constructor


def register_concentrations():
    return {k: Concentration(k) for k in COMPOUNDS}

from typing import Dict

import numpy as np
from PySDM.attributes.impl.attribute import Attribute
from PySDM.attributes.impl.extensive_attribute import ExtensiveAttribute


class Particles:

    def __init__(
            self, core,
            idx,
            extensive_attributes,
            extensive_keys: dict,
            cell_start,
            attributes: Dict[str, Attribute]
    ):
        self.core = core

        self.__n_sd = core.n_sd
        self.__valid_n_sd = core.n_sd
        self.healthy = True
        self.__healthy_memory = self.core.Storage.from_ndarray(np.full((1,), 1))
        self.__idx = idx
        self.__strides = self.core.Storage.from_ndarray(self.core.mesh.strides)

        self.extensive_attributes = extensive_attributes
        self.extensive_keys = extensive_keys

        self.cell_idx = self.core.Index.identity_index(len(cell_start) - 1)
        self.__cell_start = self.core.Storage.from_ndarray(cell_start)
        self.__cell_caretaker = self.core.bck.make_cell_caretaker(self.__idx, self.__cell_start,
                                                                  scheme=core.sorting_scheme)
        self.__sorted = False
        self.attributes = attributes

        self.recalculate_cell_id()

    @property
    def cell_start(self):
        if not self.__sorted:
            self.__sort_by_cell_id()
        return self.__cell_start

    @property
    def SD_num(self):
        assert self.healthy
        return len(self.__idx)

    def sanitize(self):
        if not self.healthy:
            self.__idx.length = self.__valid_n_sd
            self.__idx.remove_zero_n_or_flagged(self['n'])
            self.__valid_n_sd = self.__idx.length
            self.healthy = True
            self.__healthy_memory[:] = 1
            self.__sorted = False

    def cut_working_length(self, length):
        assert length <= len(self.__idx)
        self.__idx.length = length

    def get_working_length(self):
        return len(self.__idx)

    def reset_working_length(self):
        self.__idx.length = self.__valid_n_sd

    def reset_cell_idx(self):
        self.cell_idx.reset_index()
        self.__sort_by_cell_id()

    def __getitem__(self, item):
        return self.attributes[item].get()

    def permutation(self, u01, local):
        if local:
            """
            apply Fisher-Yates algorithm per cell
            """
            self.__idx.shuffle(u01, parts=self.cell_start)
        else:
            """
            apply Fisher-Yates algorithm to all super-droplets
            """
            self.__idx.shuffle(u01)
            self.__sorted = False

    def __sort_by_cell_id(self):
        self.__cell_caretaker(self['cell id'], self.cell_idx, self.__cell_start, self.__idx)
        self.__sorted = True

    def get_extensive_attrs(self):
        return self.extensive_attributes

    def recalculate_cell_id(self):
        if 'cell origin' not in self.attributes:
            return
        else:
            self.core.bck.cell_id(self['cell id'], self['cell origin'], self.__strides)
            self.__sorted = False

    def sort_within_pair_by_attr(self, is_first_in_pair, attr_name):
        self.core.bck.sort_within_pair_by_attr(self.__idx, is_first_in_pair, self[attr_name])

    def moments(self, moment_0, moments, specs: dict, attr_name='volume', attr_range=(-np.inf, np.inf),
                weighting_attribute='volume', weighting_rank=0):
        attr_data, ranks = [], []
        for attr in specs:
            for rank in specs[attr]:
                attr_data.append(self.attributes[attr].get())
                ranks.append(rank)
        assert len(set(attr_data)) <= 1
        if len(attr_data) == 0:
            attr_data = self.core.backend.Storage.empty((0,), dtype=float)
        else:
            attr_data = attr_data[0]

        ranks = self.core.bck.Storage.from_ndarray(np.array(ranks, dtype=float))

        self.core.bck.moments(moment_0,
                              moments,
                              self['n'],
                              attr_data,
                              self['cell id'],
                              self.__idx,
                              self.SD_num,
                              ranks,
                              attr_range[0], attr_range[1],
                              self[attr_name],
                              weighting_attribute=self[weighting_attribute],
                              weighting_rank=weighting_rank
                              )

    def spectrum_moments(self, moment_0, moments, attr, rank, attr_bins, attr_name='volume',
                         weighting_attribute='volume', weighting_rank=0):
        attr_data = self.attributes[attr].get()
        self.core.bck.spectrum_moments(moment_0,
                                       moments,
                                       self['n'],
                                       attr_data,
                                       self['cell id'],
                                       self.__idx,
                                       self.SD_num,
                                       rank,
                                       attr_bins,
                                       self[attr_name],
                                       weighting_attribute=self[weighting_attribute],
                                       weighting_rank=weighting_rank
                                       )

    def coalescence(self, gamma, is_first_in_pair):
        self.core.bck.coalescence(n=self['n'],
                                  idx=self.__idx,
                                  attributes=self.get_extensive_attrs(),
                                  gamma=gamma,
                                  healthy=self.__healthy_memory,
                                  is_first_in_pair=is_first_in_pair
                                  )
        self.healthy = bool(self.__healthy_memory)
        self.core.particles.sanitize()
        self.attributes['n'].mark_updated()
        for attr in self.attributes.values():
            if isinstance(attr, ExtensiveAttribute):
                attr.mark_updated()

    def adaptive_sdm_end(self, dt_left):
        return self.core.bck.adaptive_sdm_end(dt_left, self.core.particles.cell_start)

    def has_attribute(self, attr):
        return attr in self.attributes

    def remove_precipitated(self) -> float:
        res = self.core.bck.flag_precipitated(self['cell origin'], self['position in cell'],
                                              self['volume'], self['n'],
                                              self.__idx, self.SD_num, self.__healthy_memory)
        self.healthy = bool(self.__healthy_memory)
        self.sanitize()
        return res

    def oxidation(self, kinetic_consts, dt, equilibrium_consts, dissociation_factors, do_chemistry_flag):
        self.core.bck.oxidation(
            n_sd=self.core.n_sd,
            cell_ids=self['cell id'],
            do_chemistry_flag=do_chemistry_flag,
            k0=kinetic_consts["k0"],
            k1=kinetic_consts["k1"],
            k2=kinetic_consts["k2"],
            k3=kinetic_consts["k3"],
            K_SO2=equilibrium_consts["K_SO2"],
            K_HSO3=equilibrium_consts["K_HSO3"],
            dissociation_factor_SO2=dissociation_factors['SO2'],
            dt=dt,
            # input
            droplet_volume=self["volume"],
            pH=self["pH"],
            # output
            moles_O3=self["moles_O3"],
            moles_H2O2=self["moles_H2O2"],
            moles_S_IV=self["moles_S_IV"],
            moles_S_VI=self["moles_S_VI"]
        )
        self.attributes['moles_S_IV'].mark_updated()
        self.attributes['moles_S_VI'].mark_updated()
        self.attributes['moles_H2O2'].mark_updated()
        self.attributes['moles_O3'].mark_updated()

    def dissolution(self, gaseous_compounds, system_type, dissociation_factors, dt,
                    environment_mixing_ratios, do_chemistry_flag):
        self.core.bck.dissolution(
            n_cell=self.core.mesh.n_cell,
            n_threads=1,
            cell_order=np.arange(self.core.mesh.n_cell),
            cell_start_arg=self.cell_start,
            idx=self._Particles__idx,
            do_chemistry_flag=do_chemistry_flag,
            mole_amounts={key: self["moles_" + key] for key in gaseous_compounds.keys()},
            env_mixing_ratio=environment_mixing_ratios,
            # note: assuming condensation was called
            env_p=self.core.env.get_predicted('p'),
            env_T=self.core.env.get_predicted('T'),
            env_rho_d=self.core.env.get_predicted('rhod'),
            dt=dt,
            dv=self.core.mesh.dv,
            droplet_volume=self["volume"],
            multiplicity=self["n"],
            system_type=system_type,
            dissociation_factors=dissociation_factors
        )
        for key in gaseous_compounds.keys():
            self.attributes[f'moles_{key}'].mark_updated()

    def chem_recalculate_cell_data(self, equilibrium_consts, kinetic_consts):
        self.core.bck.chem_recalculate_cell_data(
            equilibrium_consts=equilibrium_consts,
            kinetic_consts=kinetic_consts,
            T=self.core.env.get_predicted('T')
        )

    def chem_recalculate_drop_data(self, dissociation_factors, equilibrium_consts):
        self.core.bck.chem_recalculate_drop_data(
            dissociation_factors=dissociation_factors,
            equilibrium_consts=equilibrium_consts,
            pH=self['pH'],
            cell_id=self['cell id']
        )

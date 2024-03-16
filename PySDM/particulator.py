"""
The very class exposing `PySDM.particulator.Particulator.run()` method for launching simulations
"""

import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_common.index import make_Index
from PySDM.backends.impl_common.indexed_storage import make_IndexedStorage
from PySDM.backends.impl_common.pair_indicator import make_PairIndicator
from PySDM.backends.impl_common.pairwise_storage import make_PairwiseStorage
from PySDM.impl.particle_attributes import ParticleAttributes


class Particulator:  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    def __init__(self, n_sd, backend: BackendMethods):
        assert isinstance(backend, BackendMethods)
        self.__n_sd = n_sd

        self.backend = backend
        self.formulae = backend.formulae
        self.environment = None
        self.attributes: (ParticleAttributes, None) = None
        self.dynamics = {}
        self.products = {}
        self.observers = []

        self.n_steps = 0

        self.sorting_scheme = "default"
        self.condensation_solver = None

        self.Index = make_Index(backend)  # pylint: disable=invalid-name
        self.PairIndicator = make_PairIndicator(backend)  # pylint: disable=invalid-name
        self.PairwiseStorage = make_PairwiseStorage(  # pylint: disable=invalid-name
            backend
        )
        self.IndexedStorage = make_IndexedStorage(  # pylint: disable=invalid-name
            backend
        )

        self.timers = {}
        self.null = self.Storage.empty(0, dtype=float)

    def run(self, steps):
        for _ in range(steps):
            for key, dynamic in self.dynamics.items():
                with self.timers[key]:
                    dynamic()
            self.n_steps += 1
            self._notify_observers()

    def _notify_observers(self):
        reversed_order_so_that_environment_is_last = reversed(self.observers)
        for observer in reversed_order_so_that_environment_is_last:
            observer.notify()

    @property
    def Storage(self):
        return self.backend.Storage

    @property
    def Random(self):
        return self.backend.Random

    @property
    def n_sd(self) -> int:
        return self.__n_sd

    @property
    def dt(self) -> float:
        if self.environment is not None:
            return self.environment.dt
        return None

    @property
    def mesh(self):
        if self.environment is not None:
            return self.environment.mesh
        return None

    def normalize(self, prob, norm_factor):
        self.backend.normalize(
            prob=prob,
            cell_id=self.attributes["cell id"],
            cell_idx=self.attributes.cell_idx,
            cell_start=self.attributes.cell_start,
            norm_factor=norm_factor,
            timestep=self.dt,
            dv=self.mesh.dv,
        )

    def update_TpRH(self):
        self.backend.temperature_pressure_rh(
            # input
            rhod=self.environment.get_predicted("rhod"),
            thd=self.environment.get_predicted("thd"),
            water_vapour_mixing_ratio=self.environment.get_predicted(
                "water_vapour_mixing_ratio"
            ),
            # output
            T=self.environment.get_predicted("T"),
            p=self.environment.get_predicted("p"),
            RH=self.environment.get_predicted("RH"),
        )

    def condensation(self, *, rtol_x, rtol_thd, counters, RH_max, success, cell_order):
        """Updates droplet volumes by simulating condensation driven by prior changes
          in environment thermodynamic state, updates the environment state.
        In the case of parcel environment, condensation is driven solely by changes in
          the dry-air density (theta and water_vapour_mixing_ratio should not be changed
          by other dynamics).
        In the case of prescribed-flow/kinematic environments, the dry-air density is
          constant in time throughout the simulation.
        This function should only change environment's predicted `thd` and
          `water_vapour_mixing_ratio` (and not `rhod`).
        """
        self.backend.condensation(
            solver=self.condensation_solver,
            n_cell=self.mesh.n_cell,
            cell_start_arg=self.attributes.cell_start,
            water_mass=self.attributes["water mass"],
            multiplicity=self.attributes["multiplicity"],
            vdry=self.attributes["dry volume"],
            idx=self.attributes._ParticleAttributes__idx,
            rhod=self.environment["rhod"],
            thd=self.environment["thd"],
            water_vapour_mixing_ratio=self.environment["water_vapour_mixing_ratio"],
            dv=self.environment.dv,
            prhod=self.environment.get_predicted("rhod"),
            pthd=self.environment.get_predicted("thd"),
            predicted_water_vapour_mixing_ratio=self.environment.get_predicted(
                "water_vapour_mixing_ratio"
            ),
            kappa=self.attributes["kappa"],
            f_org=self.attributes["dry volume organic fraction"],
            rtol_x=rtol_x,
            rtol_thd=rtol_thd,
            v_cr=self.attributes["critical volume"],
            timestep=self.dt,
            counters=counters,
            cell_order=cell_order,
            RH_max=RH_max,
            success=success,
            cell_id=self.attributes["cell id"],
            reynolds_number=self.attributes["Reynolds number"],
            air_density=self.environment["air density"],
            air_dynamic_viscosity=self.environment["air dynamic viscosity"],
        )
        self.attributes.mark_updated("water mass")

    def collision_coalescence_breakup(
        self,
        *,
        enable_breakup,
        gamma,
        rand,
        Ec,
        Eb,
        fragment_mass,
        coalescence_rate,
        breakup_rate,
        breakup_rate_deficit,
        is_first_in_pair,
        warn_overflows,
        max_multiplicity,
    ):
        # pylint: disable=too-many-locals
        idx = self.attributes._ParticleAttributes__idx
        healthy = self.attributes._ParticleAttributes__healthy_memory
        cell_id = self.attributes["cell id"]
        multiplicity = self.attributes["multiplicity"]
        attributes = self.attributes.get_extensive_attribute_storage()
        if enable_breakup:
            self.backend.collision_coalescence_breakup(
                multiplicity=multiplicity,
                idx=idx,
                attributes=attributes,
                gamma=gamma,
                rand=rand,
                Ec=Ec,
                Eb=Eb,
                fragment_mass=fragment_mass,
                healthy=healthy,
                cell_id=cell_id,
                coalescence_rate=coalescence_rate,
                breakup_rate=breakup_rate,
                breakup_rate_deficit=breakup_rate_deficit,
                is_first_in_pair=is_first_in_pair,
                warn_overflows=warn_overflows,
                particle_mass=self.attributes["water mass"],
                max_multiplicity=max_multiplicity,
            )
        else:
            self.backend.collision_coalescence(
                multiplicity=multiplicity,
                idx=idx,
                attributes=attributes,
                gamma=gamma,
                healthy=healthy,
                cell_id=cell_id,
                coalescence_rate=coalescence_rate,
                is_first_in_pair=is_first_in_pair,
            )
        self.attributes.healthy = bool(
            self.attributes._ParticleAttributes__healthy_memory
        )
        self.attributes.sanitize()
        self.attributes.mark_updated("multiplicity")
        for key in self.attributes.get_extensive_attribute_keys():
            self.attributes.mark_updated(key)

    def oxidation(
        self,
        *,
        kinetic_consts,
        timestep,
        equilibrium_consts,
        dissociation_factors,
        do_chemistry_flag,
    ):
        self.backend.oxidation(
            n_sd=self.n_sd,
            cell_ids=self.attributes["cell id"],
            do_chemistry_flag=do_chemistry_flag,
            k0=kinetic_consts["k0"],
            k1=kinetic_consts["k1"],
            k2=kinetic_consts["k2"],
            k3=kinetic_consts["k3"],
            K_SO2=equilibrium_consts["K_SO2"],
            K_HSO3=equilibrium_consts["K_HSO3"],
            dissociation_factor_SO2=dissociation_factors["SO2"],
            timestep=timestep,
            # input
            droplet_volume=self.attributes["volume"],
            pH=self.attributes["pH"],
            # output
            moles_O3=self.attributes["moles_O3"],
            moles_H2O2=self.attributes["moles_H2O2"],
            moles_S_IV=self.attributes["moles_S_IV"],
            moles_S_VI=self.attributes["moles_S_VI"],
        )
        for attr in ("moles_S_IV", "moles_S_VI", "moles_H2O2", "moles_O3"):
            self.attributes.mark_updated(attr)

    def dissolution(
        self,
        *,
        gaseous_compounds,
        system_type,
        dissociation_factors,
        timestep,
        environment_mixing_ratios,
        do_chemistry_flag,
    ):
        self.backend.dissolution(
            n_cell=self.mesh.n_cell,
            n_threads=1,
            cell_order=np.arange(self.mesh.n_cell),
            cell_start_arg=self.attributes.cell_start,
            idx=self.attributes._ParticleAttributes__idx,
            do_chemistry_flag=do_chemistry_flag,
            mole_amounts={
                key: self.attributes["moles_" + key] for key in gaseous_compounds.keys()
            },
            env_mixing_ratio=environment_mixing_ratios,
            # note: assuming condensation was called
            env_p=self.environment.get_predicted("p"),
            env_T=self.environment.get_predicted("T"),
            env_rho_d=self.environment.get_predicted("rhod"),
            timestep=timestep,
            dv=self.mesh.dv,
            droplet_volume=self.attributes["volume"],
            multiplicity=self.attributes["multiplicity"],
            system_type=system_type,
            dissociation_factors=dissociation_factors,
        )
        for key in gaseous_compounds.keys():
            self.attributes.mark_updated(f"moles_{key}")

    def chem_recalculate_cell_data(self, *, equilibrium_consts, kinetic_consts):
        self.backend.chem_recalculate_cell_data(
            equilibrium_consts=equilibrium_consts,
            kinetic_consts=kinetic_consts,
            temperature=self.environment.get_predicted("T"),
        )

    def chem_recalculate_drop_data(self, *, dissociation_factors, equilibrium_consts):
        self.backend.chem_recalculate_drop_data(
            dissociation_factors=dissociation_factors,
            equilibrium_consts=equilibrium_consts,
            pH=self.attributes["pH"],
            cell_id=self.attributes["cell id"],
        )

    def recalculate_cell_id(self):
        if not self.attributes.has_attribute("cell origin"):
            return
        self.backend.cell_id(
            self.attributes["cell id"],
            self.attributes["cell origin"],
            self.backend.Storage.from_ndarray(self.environment.mesh.strides),
        )
        self.attributes._ParticleAttributes__sorted = False

    def sort_within_pair_by_attr(self, is_first_in_pair, attr_name):
        self.backend.sort_within_pair_by_attr(
            self.attributes._ParticleAttributes__idx,
            is_first_in_pair,
            self.attributes[attr_name],
        )

    def moments(
        self,
        *,
        moment_0,
        moments,
        specs: dict,
        attr_name="water mass",
        attr_range=(-np.inf, np.inf),
        weighting_attribute="water mass",
        weighting_rank=0,
        skip_division_by_m0=False,
    ):
        """
        Writes to `moment_0` and `moment` the zero-th and the k-th statistical moments
        of particle attributes computed filtering by value of the attribute `attr_name`
        to fall within `attr_range`. The moment ranks are defined by `specs`.

        Parameters:
            specs: e.g., `specs={'volume': (1,2,3), 'kappa': (1)}` computes three moments
                of volume and one moment of kappa
            skip_division_by_m0: if set to `True`, the values written to `moments` are
                multiplied by the 0-th moment (e.g., total volume instead of mean volume)
        """
        if len(specs) == 0:
            raise ValueError("empty specs passed")
        attr_data, ranks = [], []
        for attr in specs:
            for rank in specs[attr]:
                attr_data.append(self.attributes[attr])
                ranks.append(rank)
        assert len(set(attr_data)) <= 1
        if len(attr_data) == 0:
            attr_data = self.backend.Storage.empty((0,), dtype=float)
        else:
            attr_data = attr_data[0]

        ranks = self.backend.Storage.from_ndarray(np.array(ranks, dtype=float))

        self.backend.moments(
            moment_0=moment_0,
            moments=moments,
            multiplicity=self.attributes["multiplicity"],
            attr_data=attr_data,
            cell_id=self.attributes["cell id"],
            idx=self.attributes._ParticleAttributes__idx,
            length=self.attributes.super_droplet_count,
            ranks=ranks,
            min_x=attr_range[0],
            max_x=attr_range[1],
            x_attr=self.attributes[attr_name],
            weighting_attribute=self.attributes[weighting_attribute],
            weighting_rank=weighting_rank,
            skip_division_by_m0=skip_division_by_m0,
        )

    def spectrum_moments(
        self,
        *,
        moment_0,
        moments,
        attr,
        rank,
        attr_bins,
        attr_name="water mass",
        weighting_attribute="water mass",
        weighting_rank=0,
    ):
        attr_data = self.attributes[attr]
        self.backend.spectrum_moments(
            moment_0=moment_0,
            moments=moments,
            multiplicity=self.attributes["multiplicity"],
            attr_data=attr_data,
            cell_id=self.attributes["cell id"],
            idx=self.attributes._ParticleAttributes__idx,
            length=self.attributes.super_droplet_count,
            rank=rank,
            x_bins=attr_bins,
            x_attr=self.attributes[attr_name],
            weighting_attribute=self.attributes[weighting_attribute],
            weighting_rank=weighting_rank,
        )

    def adaptive_sdm_end(self, dt_left):
        return self.backend.adaptive_sdm_end(dt_left, self.attributes.cell_start)

    def remove_precipitated(
        self, *, displacement, precipitation_counting_level_index
    ) -> float:
        res = self.backend.flag_precipitated(
            cell_origin=self.attributes["cell origin"],
            position_in_cell=self.attributes["position in cell"],
            volume=self.attributes["volume"],
            multiplicity=self.attributes["multiplicity"],
            idx=self.attributes._ParticleAttributes__idx,
            length=self.attributes.super_droplet_count,
            healthy=self.attributes._ParticleAttributes__healthy_memory,
            precipitation_counting_level_index=precipitation_counting_level_index,
            displacement=displacement,
        )
        self.attributes.healthy = bool(
            self.attributes._ParticleAttributes__healthy_memory
        )
        self.attributes.sanitize()
        return res

    def flag_out_of_column(self):
        self.backend.flag_out_of_column(
            cell_origin=self.attributes["cell origin"],
            position_in_cell=self.attributes["position in cell"],
            idx=self.attributes._ParticleAttributes__idx,
            length=self.attributes.super_droplet_count,
            healthy=self.attributes._ParticleAttributes__healthy_memory,
            domain_top_level_index=self.mesh.grid[-1],
        )
        self.attributes.healthy = bool(
            self.attributes._ParticleAttributes__healthy_memory
        )
        self.attributes.sanitize()

    def calculate_displacement(
        self, *, displacement, courant, cell_origin, position_in_cell, n_substeps
    ):
        for dim in range(len(self.environment.mesh.grid)):
            self.backend.calculate_displacement(
                dim=dim,
                displacement=displacement,
                courant=courant[dim],
                cell_origin=cell_origin,
                position_in_cell=position_in_cell,
                n_substeps=n_substeps,
            )

    def isotopic_fractionation(self, heavy_isotopes: tuple):
        self.backend.isotopic_fractionation()
        for isotope in heavy_isotopes:
            self.attributes.mark_updated(f"moles_{isotope}")

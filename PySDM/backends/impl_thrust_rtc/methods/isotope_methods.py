"""
GPU implementation of isotope-relates backend methods
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class IsotopeMethods(ThrustRTCBackendMethods):
    @cached_property
    def __isotopic_delta(self):
        return trtc.For(
            param_names=("output", "ratio", "reference_ratio"),
            name_iter="i",
            body=f"""
            output[i] = {self.formulae.trivia.isotopic_ratio_2_delta.c_inline(
                ratio="ratio[i]",
                reference_ratio="reference_ratio"
            )};
            """.replace("real_type", self._get_c_type()),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def isotopic_delta(self, output, ratio, reference_ratio):
        self.__isotopic_delta.launch_n(
            n=output.shape[0],
            args=(output.data, ratio.data, self._get_floating_point(reference_ratio)),
        )

    @cached_property
    def __isotopic_fractionation(self):
        return trtc.For(
            param_names=(
                "cell_id",
                "cell_volume",
                "multiplicity",
                "dm_total",
                "signed_water_mass",
                "dry_air_density",
                "molar_mass_heavy_molecule",
                "moles_heavy_molecule",
                "bolin_number",
                "molality_in_dry_air",
            ),
            name_iter="i",
            body="""
            auto cid = cell_id[i];

            real_type mass_ratio = (moles_heavy_molecule[i] * molar_mass_heavy_molecule) / signed_water_mass[i];

            real_type dm_heavy = 0;

            if (bolin_number[i] != 0) {
                dm_heavy = dm_total[i] / bolin_number[i] * mass_ratio;
            }

            real_type dn_heavy = dm_heavy / molar_mass_heavy_molecule;

            moles_heavy_molecule[i] += dn_heavy;

            real_type mass_of_dry_air = dry_air_density[cid] * cell_volume[cid];

            atomicAdd((real_type*) &molality_in_dry_air[cid], -dn_heavy * multiplicity[i] / mass_of_dry_air);
            """.replace("real_type", self._get_c_type()),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def isotopic_fractionation(
        self,
        cell_id,
        cell_volume,
        multiplicity,
        dm_total,
        signed_water_mass,
        dry_air_density,
        molar_mass_heavy_molecule,
        moles_heavy_molecule,
        bolin_number,
        molality_in_dry_air,
    ):  # pylint: disable=too-many-positional-arguments
        self.__isotopic_fractionation.launch_n(
            n=len(multiplicity),
            args=(
                cell_id.data,
                cell_volume.data,
                multiplicity.data,
                dm_total.data,
                signed_water_mass.data,
                dry_air_density.data,
                self._get_floating_point(molar_mass_heavy_molecule),
                moles_heavy_molecule.data,
                bolin_number.data,
                molality_in_dry_air.data,
            ),
        )

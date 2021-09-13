from PySDM.backends import CPU
from PySDM import Builder
from PySDM.environments import Box
from PySDM.dynamics import Freezing
from PySDM.physics import si, constants as const, Formulae
from PySDM.initialisation.multiplicities import discretise_n
from PySDM.initialisation import spectral_sampling
from PySDM.physics.spectra import Lognormal
from PySDM.products import IceWaterContent
import numpy as np
from matplotlib import pylab
import pytest

n_runs_per_case = 3

@pytest.mark.parametrize("multiplicity", (1, 2, 10))
def test_AK16_fig_1(multiplicity, plot=False):
    # Arrange
    J_het = 1e3 / si.cm ** 2 / si.s
    A_g = 1e-5 * si.cm ** 2

    dt = 1 * si.s
    total_time = 6 * si.min

    # dummy multipliers (multiplied and then divided by)
    dv = 1 * si.cm ** 3  # will become used if coalescence or other processes are turned on
    droplet_volume = 1 * si.um ** 3  # ditto

    cases = {
        'Iso3': {'ISA': Lognormal(norm_factor=1000 / dv, m_mode=A_g, s_geom=10), 'color': '#1A62B4'},
        'Iso4': {'ISA': Lognormal(norm_factor=30 / dv, m_mode=A_g, s_geom=10), 'color': '#95BDE1'},
        'Iso1': {'ISA': Lognormal(norm_factor=1000 / dv, m_mode=A_g, s_geom=1), 'color': '#298131'},
        'Iso2': {'ISA': Lognormal(norm_factor=30 / dv, m_mode=A_g, s_geom=1), 'color': '#9ACFA4'},
    }

    # Act
    output = {}

    for key, case in cases.items():
        output[key] = []
        for i in range(n_runs_per_case):
            seed = i
            number_of_real_droplets = case['ISA'].norm_factor * dv
            n_sd = number_of_real_droplets / multiplicity
            assert int(n_sd) == n_sd
            n_sd = int(n_sd)

            formulae = Formulae(seed=seed)
            builder = Builder(n_sd=n_sd, backend=CPU, formulae=formulae)
            builder.set_environment(Box(dt=dt, dv=dv))
            builder.add_dynamic(Freezing(singular=False, J_het=J_het))

            if case['ISA'].s_geom != 1:
                _isa, _conc = spectral_sampling.ConstantMultiplicity(case['ISA']).sample(n_sd)
            else:
                _isa, _conc = np.full(n_sd, A_g), np.full(n_sd, multiplicity / dv)
            attributes = {
                'n': discretise_n(_conc * dv),
                'immersed surface area': _isa,
                'volume': np.full(n_sd, droplet_volume)
            }
            np.testing.assert_almost_equal(attributes['n'], multiplicity)
            products = (IceWaterContent(specific=False),)
            particulator = builder.build(attributes=attributes, products=products)

            cell_id = 0
            data = []
            for i in range(int(total_time / dt) + 1):
                particulator.run(0 if i == 0 else 1)

                ice_mass_per_volume = particulator.products['qi'].get()[cell_id]
                ice_mass = ice_mass_per_volume * dv
                ice_number = ice_mass / (const.rho_w * droplet_volume)
                unfrozen_fraction = 1 - ice_number / number_of_real_droplets
                data.append(unfrozen_fraction)

            output[key].append(data)

    # Plot
    pylab.rc('font', size=10)
    for key in output.keys():
        for run in range(n_runs_per_case):
            label = f"{key}: σ=ln({int(cases[key]['ISA'].s_geom)}),N={int(cases[key]['ISA'].norm_factor * dv)}"
            pylab.step(
                dt / si.min * np.arange(len(output[key][run])),
                output[key][run],
                label=label if run == 0 else None,
                color=cases[key]['color'],
                linewidth=.666
            )
        output[key].append(np.mean(np.asarray(output[key]), axis=0))
        pylab.step(
            dt / si.min * np.arange(len(output[key][-1])),
            output[key][-1],
            color=cases[key]['color'],
            linewidth=1.666
        )

    pylab.legend()
    pylab.yscale('log')
    pylab.ylim(1e-2, 1)
    pylab.xlim(0, total_time / si.min)
    pylab.xlabel("t / min")
    pylab.ylabel("$f_{ufz}$")
    pylab.gca().set_aspect(3)
    pylab.show()

    # Assert
    np.testing.assert_array_less(
        output['Iso3'][-1][1:int(1 * si.min / dt)],
        output['Iso1'][-1][1:int(1 * si.min / dt)]
    )
    np.testing.assert_array_less(
        output['Iso1'][-1][int(2 * si.min / dt):],
        output['Iso3'][-1][int(2 * si.min / dt):]
    )
    np.testing.assert_array_less(
        output['Iso2'][int(.5 * si.min / dt):],
        output['Iso1'][int(.5 * si.min / dt):]
    )

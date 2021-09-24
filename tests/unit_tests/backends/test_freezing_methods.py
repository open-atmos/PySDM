from PySDM.physics import constants as const, Formulae
from PySDM.physics.heterogeneous_ice_nucleation_rate import constant
from PySDM import Builder
from PySDM.backends import CPU
from PySDM.environments import Box
from PySDM.dynamics import Freezing
from PySDM.products import IceWaterContent

# noinspection PyUnresolvedReferences
from ...backends_fixture import backend  # TODO #599

from matplotlib import pylab
import numpy as np


class TestFreezingMethods:
    # TODO #599
    # @staticmethod
    # def test_freeze_singular(backend):
    #     pass

    @staticmethod
    def test_freeze_time_dependent(plot=False):
        # Arrange
        cases = (
            {'dt': 5e5, 'N':  1},
            {'dt': 1e6, 'N':  1},
            {'dt': 5e5, 'N':  8},
            {'dt': 1e6, 'N':  8},
            {'dt': 5e5, 'N': 32},
            {'dt': 1e6, 'N': 32},
        )
        rate = 1e-9
        immersed_surface_area = 1
        constant.J_het = rate / immersed_surface_area

        number_of_real_droplets = 1024
        total_time = 2e9  # effectively interpretted here as seconds, i.e. cycle = 1 * si.s

        # dummy (but must-be-set) values
        vol = 44  # just to enable sign flipping (ice water uses negative volumes), actual value does not matter
        dv = 666  # products use concentration, just dividing there and multiplying back here, actual value does not matter

        hgh = lambda t: np.exp(-0.8 * rate * (t - total_time / 10))
        low = lambda t: np.exp(-1.2 * rate * (t + total_time / 10))

        # Act
        output = {}

        for case in cases:
            n_sd = int(number_of_real_droplets // case['N'])
            assert n_sd == number_of_real_droplets / case['N']
            assert total_time // case['dt'] == total_time / case['dt']

            key = f"{case['dt']}:{case['N']}"
            output[key] = {'unfrozen_fraction': [], 'dt': case['dt'], 'N': case['N']}

            formulae = Formulae(heterogeneous_ice_nucleation_rate='Constant')
            builder = Builder(n_sd=n_sd, backend=CPU(formulae=formulae))
            env = Box(dt=case['dt'], dv=dv)
            builder.set_environment(env)
            builder.add_dynamic(Freezing(singular=False))
            attributes = {
                'n': np.full(n_sd, int(case['N'])),
                'immersed surface area': np.full(n_sd, immersed_surface_area),
                'volume': np.full(n_sd, vol)
            }
            products = (IceWaterContent(specific=False),)
            particulator = builder.build(attributes=attributes, products=products)

            env['a_w_ice'] = np.nan

            cell_id = 0
            for i in range(int(total_time / case['dt']) + 1):
                particulator.run(0 if i == 0 else 1)

                ice_mass_per_volume = particulator.products['qi'].get()[cell_id]
                ice_mass = ice_mass_per_volume * dv
                ice_number = ice_mass / (const.rho_w * vol)
                unfrozen_fraction = 1 - ice_number / number_of_real_droplets
                output[key]['unfrozen_fraction'].append(unfrozen_fraction)

        # Plot
        if plot:
            fit_x = np.linspace(0, total_time, num=100)
            fit_y = np.exp(-rate * fit_x)

            for key in output.keys():
                pylab.step(
                    output[key]['dt'] * np.arange(len(output[key]['unfrozen_fraction'])),
                    output[key]['unfrozen_fraction'],
                    label=f"dt={output[key]['dt']:.2g} / N={output[key]['N']}",
                    marker='.',
                    linewidth=1 + output[key]['N']//8
                )

            pylab.plot(fit_x, fit_y, color='black', linestyle='--', label='theory', linewidth=5)
            pylab.plot(fit_x, hgh(fit_x), color='black', linestyle=':', label='assert upper bound')
            pylab.plot(fit_x, low(fit_x), color='black', linestyle=':', label='assert lower bound')
            pylab.legend()
            pylab.yscale('log')
            pylab.ylim(fit_y[-1], fit_y[0])
            pylab.xlim(0, total_time)
            pylab.xlabel("time")
            pylab.ylabel("unfrozen fraction")
            pylab.grid()
            pylab.show()

        # Assert
        for key in output.keys():
            data = np.asarray(output[key]['unfrozen_fraction'])
            x = output[key]['dt'] * np.arange(len(data))
            np.testing.assert_array_less(data, hgh(x))
            np.testing.assert_array_less(low(x), data)

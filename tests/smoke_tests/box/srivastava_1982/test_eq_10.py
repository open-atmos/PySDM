# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from matplotlib import pyplot
from PySDM_examples.Srivastava_1982.equations import Equations, EquationsHelpers
from PySDM_examples.Srivastava_1982.example import (
    add_to_plot_simulation_results,
    compute_log_space,
    get_coalescence_analytic_results,
    get_processed_results,
    get_pysdm_secondary_products,
)
from PySDM_examples.Srivastava_1982.settings import Settings, SimProducts
from PySDM_examples.Srivastava_1982.simulation import Simulation

from PySDM.dynamics import Coalescence
from PySDM.dynamics.collisions.collision_kernels import ConstantK
from PySDM.physics import si

ASSERT_PROD = SimProducts.Computed.mean_drop_volume_total_volume_ratio.name
N_STEPS = 32
N_REALISATIONS = 5
SEEDS = list(range(N_REALISATIONS))


def test_pysdm_coalescence_is_close_to_analytic_coalescence(
    plot=False,
):  # TODO #987 (backend_class: CPU, GPU)
    # arrange
    settings = Settings(
        srivastava_c=0.5e-6 / si.s,
        frag_mass=-1 * si.g,
        drop_mass_0=1 * si.g,
        dt=1 * si.s,
        dv=1 * si.m**3,
        n_sds=(16, 128),
        total_number=1e6,
    )

    simulation = Simulation(
        n_steps=N_STEPS,
        settings=settings,
        collision_dynamic=Coalescence(
            collision_kernel=ConstantK(a=settings.srivastava_c)
        ),
    )

    x = np.arange(N_STEPS + 1, dtype=float)

    equations = Equations(
        M=settings.total_volume * settings.rho / settings.frag_mass,
        c=settings.srivastava_c,
    )
    equation_helper = EquationsHelpers(
        settings.total_volume,
        settings.total_number_0,
        settings.rho,
        frag_mass=settings.frag_mass,
    )
    m0 = equation_helper.m0()

    x_log = compute_log_space(x)
    analytic_results = get_coalescence_analytic_results(
        equations, settings, m0, x, x_log
    )

    # act
    sim_products = simulation.run_convergence_analysis(x, seeds=SEEDS)
    secondary_products = get_pysdm_secondary_products(
        products=sim_products, total_volume=settings.total_volume
    )

    pysdm_results = get_processed_results(secondary_products)

    plot_prods = [
        k
        for k in list(pysdm_results.values())[0].keys()
        if k != SimProducts.PySDM.total_volume.name
    ]

    # plot
    add_to_plot_simulation_results(
        plot_prods,
        settings.n_sds,
        x,
        pysdm_results,
        analytic_results,
    )

    if plot:
        pyplot.show()

    # assert
    np.testing.assert_allclose(
        actual=pysdm_results[settings.n_sds[-1]][ASSERT_PROD]["avg"],
        desired=analytic_results[ASSERT_PROD],
        rtol=2e-1,
    )
    assert np.mean(pysdm_results[settings.n_sds[-1]][ASSERT_PROD]["std"]) < np.mean(
        pysdm_results[settings.n_sds[0]][ASSERT_PROD]["std"]
    )

    assert np.mean(
        np.abs(
            pysdm_results[settings.n_sds[-1]][ASSERT_PROD]["avg"]
            - analytic_results[ASSERT_PROD]
        )
    ) < np.mean(
        np.abs(
            pysdm_results[settings.n_sds[0]][ASSERT_PROD]["avg"]
            - analytic_results[ASSERT_PROD]
        )
    )

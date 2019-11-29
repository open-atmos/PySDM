"""
Created at 22.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np

from PySDM.simulation.particles import Particles as Particles
from PySDM.simulation.dynamics.advection import Advection
from PySDM.simulation.dynamics.condensation import Condensation
from PySDM.simulation.dynamics.coalescence.algorithms.sdm import SDM
from PySDM.simulation.initialisation import spatial_discretisation, spectral_discretisation
from PySDM.simulation.environment.kinematic_2d import Kinematic2D

from examples.ICMW_2012_case_1.setup import Setup
from examples.ICMW_2012_case_1.storage import Storage
from MPyDATA.mpdata.mpdata_factory import MPDATAFactory, z_vec_coord, x_vec_coord


# instantiation of simulation components, time-stepping
def main():
    setup = Setup()
    setup.grid = (5, 5)
    setup.n_sd_per_gridbox = 20
    setup.dt /= 1
    particles = Particles(n_sd=setup.n_sd,
                               dt=setup.dt,
                               size=setup.size,
                               grid=setup.grid,
                               backend=setup.backend)

    particles.set_environment(Kinematic2D, (
        setup.stream_function,
        setup.field_values,
        setup.rhod
    ))

    particles.create_state_2d2( # TODO: ...
                                    extensive={},
                                    intensive={},
                                    spatial_discretisation=spatial_discretisation.pseudorandom,
                                    spectral_discretisation=spectral_discretisation.constant_multiplicity,
                                    spectrum_per_mass_of_dry_air=setup.spectrum_per_mass_of_dry_air,
                                    r_range=(setup.r_min, setup.r_max),
                                    kappa=setup.kappa
    )

    particles.add_dynamic(Condensation, (particles.environment, setup.kappa))
    particles.add_dynamic(Advection, ('FTBS',))

    for step in setup.steps:  # TODO: rename output_steps
        for _ in range(step - particles.n_steps):
            particles.run(1)


if __name__ == '__main__':
    main()

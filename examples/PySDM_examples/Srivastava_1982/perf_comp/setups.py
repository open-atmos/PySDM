import os
from datetime import datetime
import json
from PySDM.physics import si
from PySDM_examples.Srivastava_1982 import coalescence_and_breakup_eq13, Settings
from open_atmos_jupyter_utils import show_plot

import numpy as np
from matplotlib import pyplot
from PySDM_examples.Srivastava_1982.simulation import Simulation
import numba

from PySDM.products import SuperDropletCountPerGridbox, VolumeFirstMoment, ZerothMoment
from PySDM.backends import GPU, CPU
from PySDM.dynamics import Collision
from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import ConstantSize
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.collision_kernels import ConstantK
from .utils import ProductsNames

dt = 1 * si.s
dv = 1 * si.m**3
drop_mass_0=1 * si.g
total_number=1e12

NO_BOUNCE = ConstEb(1)


def setup_simulation(settings, n_sd, seed, double_precision=True):
  products=(
      SuperDropletCountPerGridbox(
          name=ProductsNames.super_particle_count
      ),
      VolumeFirstMoment(name=ProductsNames.total_volume),
      ZerothMoment(name=ProductsNames.total_number)
  )

  collision_rate = settings.srivastava_c + settings.srivastava_beta
  simulation = Simulation(
      n_steps=None,
      settings=settings,
      collision_dynamic=Collision(
          collision_kernel=ConstantK(a=collision_rate),
          coalescence_efficiency=ConstEc(settings.srivastava_c / collision_rate),
          breakup_efficiency=NO_BOUNCE,
          fragmentation_function=ConstantSize(c=settings.frag_mass / settings.rho),
          warn_overflows=False,
          adaptive=False,
      ),
      double_precision=double_precision,
  )
  particulator = simulation.build(n_sd, seed, products=products)

  return particulator


def setup_coalescence_only_sim(n_sd, backend_class, seed, double_precision=True):

  title = "fig_coalescence-only"
  c = 0.5e-6 / si.s
  beta = 1e-15 / si.s
  frag_mass = -1 * si.g

  settings = Settings(
      srivastava_c=c,
      srivastava_beta=beta,
      frag_mass=frag_mass,
      drop_mass_0=drop_mass_0,
      dt=dt,
      dv=dv,
      n_sds=(),
      total_number=total_number,
      backend_class=backend_class,
  )

  return setup_simulation(settings, n_sd, seed, double_precision)

def setup_breakup_only_sim(n_sd, backend_class, seed, double_precision=True):
  title = "fig_breakup-only"
  c =  1e-15 / si.s
  beta = 1e-9 / si.s
  frag_mass = 0.25 * si.g

  settings = Settings(
      srivastava_c=c,
      srivastava_beta=beta,
      frag_mass=frag_mass,
      drop_mass_0=drop_mass_0,
      dt=dt,
      dv=dv,
      n_sds=(),
      total_number=total_number,
      backend_class=backend_class,
  )
  return setup_simulation(settings, n_sd, seed, double_precision)
    

def setup_coalescence_breakup_sim(n_sd, backend_class, seed, double_precision=True):
  title = "fig_coalescence-breakup"
  c =  0.5e-6 / si.s
  beta = 1e-9 / si.s
  frag_mass = 0.25 * si.g

  settings = Settings(
      srivastava_c=c,
      srivastava_beta=beta,
      frag_mass=frag_mass,
      drop_mass_0=drop_mass_0,
      dt=dt,
      dv=dv,
      n_sds=(),
      total_number=total_number,
      backend_class=backend_class,
  )
  return setup_simulation(settings, n_sd, seed, double_precision)

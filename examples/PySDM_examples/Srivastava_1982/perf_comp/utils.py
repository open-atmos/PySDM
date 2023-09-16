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
import gc

import time
    


class ProductsNames:
  super_particle_count = "super_particle_count"
  total_volume="total_volume"
  total_number="total_number"


def print_all_products(particulator):
  print(ProductsNames.total_number, particulator.products[ProductsNames.total_number].get())
  print(ProductsNames.total_volume, particulator.products[ProductsNames.total_volume].get())
  print(ProductsNames.super_particle_count, particulator.products[ProductsNames.super_particle_count].get())


def get_prod_dict(particulator):
  d = {
    ProductsNames.total_number: list(particulator.products[ProductsNames.total_number].get()),
    ProductsNames.total_volume: list(particulator.products[ProductsNames.total_volume].get()),
    ProductsNames.super_particle_count: list(particulator.products[ProductsNames.super_particle_count].get())
  }

  return d


def measure_time_for_each_step(particulator, n_steps):
    particulator.run(steps=1)
    
    res = []
    for _ in range(n_steps):
        t0 = time.time()
        particulator.run(steps=1)
        t1 = time.time()

        res.append(t1 - t0)

    return res


def measure_time_per_timestep(particulator, n_steps):
    particulator.run(steps=1)
    
    t0 = time.time()
    particulator.run(steps=n_steps)
    t1 = time.time()

    return (t1 - t0) / n_steps


def go_benchmark(
  setup_sim, n_sds, n_steps, seeds, 
  numba_n_threads=[None], 
  double_precision=True, 
  general_filename=None,
  total_number=None,
  dv=None,
  time_measurement_fun=measure_time_per_timestep,
  backends=[CPU, GPU]
):
  TIMESTAMP = str(datetime.now()).replace(' ', '_')

  products = {}

  results = {}

  cpu_backends_configs = [(CPU, i) for i in numba_n_threads]
  backend_configs = []
  if CPU in backends:
    backend_configs = [*backend_configs, *cpu_backends_configs]
  if GPU in backends:
    backend_configs.append((GPU, None))

  for backend_class, n_threads in backend_configs:
    backend_name = backend_class.__name__
    if n_threads:
      numba.set_num_threads(n_threads)
      backend_name += "_" + str(numba.get_num_threads())
      

    results[backend_name] = {}
    products[backend_name] = {}

    print('\n', 'before')

    for n_sd in n_sds:
      print()
      print(backend_name, n_sd)
      print()
      results[backend_name][n_sd] = {}
      products[backend_name][n_sd] = {}

      for seed in seeds:

        gc.collect()

        
        particulator = setup_sim(n_sd, backend_class, seed, double_precision=double_precision, total_number=total_number, dv=dv)

        print()
        print('prod before')
        print_all_products(particulator)
        
        # particulator.run(steps=1)

        print()
        print('start')

        elapsed_time = time_measurement_fun(particulator, n_steps)

        print('\n', 'after')
        print_all_products(particulator)

        results[backend_name][n_sd][seed] = elapsed_time
        products[backend_name][n_sd][seed] = get_prod_dict(particulator)

        gc.collect()
        del particulator
        gc.collect()

  if general_filename:
    write_to_file(filename=f"{general_filename}-products", d=products)

  return results


def process_results(res_d, axis=None):
  processed_d = {}
  for backend in res_d.keys():
    processed_d[backend] = {}

    for n_sd in res_d[backend].keys():
      processed_d[backend][n_sd] = {}

      vals = res_d[backend][n_sd].values()
      vals = np.array(list(vals))

      processed_d[backend][n_sd]['mean'] = np.mean(vals, axis=axis)
      processed_d[backend][n_sd]['std'] = np.std(vals, axis=axis)
      processed_d[backend][n_sd]['max'] = np.amax(vals, axis=axis)
      processed_d[backend][n_sd]['min'] = np.amin(vals, axis=axis)

  return processed_d


def write_to_file(filename, d):
  assert not os.path.isfile(filename), filename

  with open(filename, "w") as fp:
    json.dump(d, fp)


def __plot_get_n_sd_list(backends, processed_d):
    x = []
    
    for backend in backends:
        for n_sd in processed_d[backend].keys():
          if n_sd not in x:
            x.append(n_sd)
        
    x.sort()
    return x

def __plot_get_sorted_backend_list(processed_d):
    backends = list(processed_d.keys())
    
    backends.sort()
    backends.sort(key=lambda x: int(x[6:]) if 'Numba_' in x else 100**10)

    return backends


def __plot_get_backend_markers(backends):
    markers = {backend: 'o' if 'Numba' in backend else 'x' for backend in backends}
    return markers



def plot_processed_results(processed_d, show=True, plot_label='', plot_title=None, metric='min', plot_filename=None):
  backends = __plot_get_sorted_backend_list(processed_d)

  markers = __plot_get_backend_markers(backends)

  x = __plot_get_n_sd_list(backends, processed_d)

  y = []

  for backend in backends:
    y = []
    for n_sd in x:
      v = processed_d[backend][n_sd][metric]
      assert type(v) == int or float, 'must be scalar'
      y.append(v)

    pyplot.plot(x, y, label=backend+plot_label, marker=markers[backend])

  pyplot.legend() #bbox_to_anchor =(1.1, 1))
  pyplot.xscale('log', base=2)
  pyplot.yscale('log', base=2)
  pyplot.grid()
  pyplot.xticks(x)
  pyplot.xlabel("number of super-droplets")
  pyplot.ylabel("wall time per timestep [s]")

  if plot_title:
    pyplot.title(plot_title)

  if show:
    if plot_filename:
      show_plot(filename=plot_filename)
    else:
      pyplot.show()  


def plot_processed_on_same_plot(coal_d, break_d, coal_break_d):
  plot_processed_results(coal_d, plot_label='-c', show=False)
  plot_processed_results(break_d, plot_label='-b', show=False)
  plot_processed_results(coal_break_d, plot_label='-cb', show=False)

  show_plot()



def plot_time_per_step(processed_d, n_sd, show=True, plot_label='', plot_title=None, metric='mean', plot_filename=None, step_from_to=None):
  backends = __plot_get_sorted_backend_list(processed_d)

  markers = __plot_get_backend_markers(backends)

  for backend in backends:

    y = processed_d[backend][n_sd][metric]
    x = np.arange(len(y))
      
    if step_from_to is not None:
      x = x[step_from_to[0]:step_from_to[1]]
      y = y[step_from_to[0]:step_from_to[1]]

    pyplot.plot(x, y, label=backend+plot_label, marker=markers[backend])

  pyplot.legend() #bbox_to_anchor =(1.1, 1))
  pyplot.grid()
  pyplot.xticks(x)
  pyplot.xlabel("number of super-droplets")
  pyplot.ylabel("wall time per timestep [s]")

  if plot_title:
    pyplot.title(plot_title + f'(n_sd: {n_sd})')


  if show:
    if plot_filename:
      show_plot(filename=plot_filename)
    else:
      pyplot.show()




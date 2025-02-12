from matplotlib import pyplot
from PySDM.physics import si
from PySDM.backends import CPU
from PySDM.dynamics import AmbientThermodynamics, Condensation
from PySDM.environments import Parcel
from PySDM import Builder, Formulae, products
import PySDM.initialisation.sampling.spectral_sampling  as spec_sampling  #import ConstantMultiplicity
from PySDM.initialisation.spectra.exponential import Exponential
from PySDM.products import ParticleVolumeVersusRadiusLogarithmSpectrum
import numpy as np


w_par = 1.* si.m / si.s

dt_par = 0.05 * si.m / w_par 

env = Parcel(
  dt=dt_par,    #.25 * si.s
  mass_of_dry_air=1e3 * si.kg,
  p0=200. * si.hPa,
  initial_water_vapour_mixing_ratio=20 * si.g / si.kg,
  T0=235 * si.K,
  w=w_par
)

n_sd = 100
initial_spectrum = Exponential(norm_factor=8.39e12, scale=1.19e5 * si.um ** 3)

const_mult  = spec_sampling.ConstantMultiplicity(initial_spectrum).sample(n_sd)
loga = spec_sampling.Logarithmic(initial_spectrum).sample(n_sd)
lin  = spec_sampling.Linear(initial_spectrum).sample(n_sd)


#uniform =  spec_sampling.UniformRandom(initial_spectrum).sample(n_sd)

# print( initial_spectrum.__dict__ )
# print( initial_spectrum.distribution )
# print( const_mult  )
#initial_spectrum = [ 10. * si.um ** 3, ]

formulae = Formulae()

builder = Builder(backend=CPU(formulae), n_sd=n_sd, environment=env)
builder.add_dynamic(AmbientThermodynamics())

print( const_mult[0] )
print( loga[0] )
print( lin[0] )

radius_bins_edges = np.logspace(np.log10(10 * si.um), np.log10(5e3 * si.um), num=32)
products = [ParticleVolumeVersusRadiusLogarithmSpectrum(radius_bins_edges=radius_bins_edges, name='dv/dlnr')]



# attributes = {}
# attributes['volume'], attributes['multiplicity'] = spec_sampling.UniformRandom(initial_spectrum).sample(n_sd)




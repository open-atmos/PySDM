# using Pkg
#Pkg.add("PyCall")
#Pkg.add("Plots")

# Pkg.add("Conda")
# using Conda
# Conda.pip_interop(true)
# Conda.pip("install --pre", "git+https://github.com/atmos-cloud-sim-uj/PySDM.git")

using PyCall
PySDM = pyimport("PySDM")
PySDM_backends = pyimport("PySDM.backends")
PySDM_physics_formulae = pyimport("PySDM.physics.formulae")
PySDM_physics_constants = pyimport("PySDM.physics.constants")
PySDM_environments = pyimport("PySDM.environments")
PySDM_dynamics = pyimport("PySDM.dynamics")
PySDM_initialisation_spectra = pyimport("PySDM.initialisation.spectra")
PySDM_initialisation_spectral_sampling = pyimport("PySDM.initialisation.spectral_sampling")
PySDM_dynamics_coalescence_kernels = pyimport("PySDM.dynamics.coalescence.kernels")
PySDM_products = pyimport("PySDM.products")

si = PySDM.physics.si

n_sd = 2^17
initial_spectrum = PySDM_initialisation_spectra.Exponential(
    norm_factor=8.39e12, scale=1.19e5 * si.um^3
)

builder = PySDM.Builder(backend=PySDM_backends.CPU, n_sd=n_sd)
environment = PySDM_environments.Box(dt=1 * si.s, dv=1e6 * si.m^3)
builder.set_environment(environment)
attributes = environment.init_attributes(
  spectral_discretisation=PySDM_initialisation_spectral_sampling.ConstantMultiplicity(spectrum=initial_spectrum)
)
builder.add_dynamic(PySDM_dynamics.Coalescence(kernel=PySDM_dynamics_coalescence_kernels.Golovin(b=1.5e3 / si.s)))
products = [PySDM_products.ParticlesVolumeSpectrum()] 
core = builder.build(attributes, products)

radius_bins_edges = 10 .^ range(log10(10*si.um), log10(5e3*si.um), length=32) 

using Plots
for step = 0:1200:3600
    core.run(step - core.n_steps)
    plot!(
        radius_bins_edges[1:end-1] / si.um,
        core.products["dv/dlnr"].get(radius_bins_edges) * PySDM_physics_constants.rho_w / si.g,
        linetype=:steppost,
        xaxis=:log,
        xlabel="particle radius [µm]",
        ylabel="dm/dlnr [g/m^3/(unit dr/r)]",
        label="t = $step s"
    )
end
savefig("plot.svg")

#using Pkg
#Pkg.add("PyCall")
#Pkg.add("Plots")

using PyCall
PySDM = pyimport("PySDM")
PySDM_backends = pyimport("PySDM.backends")
PySDM_physics_formulae = pyimport("PySDM.physics.formulae")
PySDM_environments = pyimport("PySDM.environments")
PySDM_dynamics = pyimport("PySDM.dynamics")
PySDM_initialisation = pyimport("PySDM.initialisation")
PySDM_initialisation_spectra = pyimport("PySDM.initialisation.spectra")
PySDM_products = pyimport("PySDM.products")

builder = PySDM.Builder(backend=PySDM_backends.CPU, n_sd=1)
si = PySDM.physics.si
environment = PySDM_environments.Parcel(
    dt=1 * si.s,
    mass_of_dry_air=1 * si.kg,
    p0=1000 * si.hPa,
    q0=20 * si.g / si.kg,
    T0=300 * si.K,
    w= 1 * si.m / si.s
)
builder.set_environment(environment)

kappa = 1 * si.dimensionless
builder.add_dynamic(PySDM_dynamics.AmbientThermodynamics())
builder.add_dynamic(PySDM_dynamics.Condensation(kappa=kappa))
        
attributes = Dict()
r_dry, specific_concentration = PySDM_initialisation.spectral_sampling.Logarithmic(
    spectrum=PySDM_initialisation_spectra.Lognormal(
        norm_factor=1000 / si.milligram,
        m_mode=50 * si.nanometre,
        s_geom=1.4 * si.dimensionless
    ),
    size_range=(10.633 * si.nanometre, 513.06 * si.nanometre)
).sample(n_sd=builder.core.n_sd)
  
attributes["dry volume"] = PySDM_physics_formulae.volume(radius=r_dry)
attributes["n"] = PySDM_initialisation.multiplicities.discretise_n(specific_concentration * environment.mass_of_dry_air)
r_wet = PySDM_initialisation.r_wet_init(r_dry, environment, zeros(Int, size(attributes["n"])), kappa)
attributes["volume"] = PySDM_physics_formulae.volume(radius=r_wet) 

products = [
  PySDM_products.ParticleMeanRadius(), 
  PySDM_products.RelativeHumidity(),
  PySDM_products.CloudConcentration(radius_range=(.5 * si.um, 25 * si.um))
]

core = builder.build(attributes, products)
    
steps = 100
output = Dict("z" => Array{Float32}(undef, steps+1))
for (_, product) in core.products
    output[product.name] = Array{Float32}(undef, steps+1)
    output[product.name][1] = product.get()[1]
end 
output["z"][1] = environment.__getitem__("z")[1]
    
for step = 2:steps+1
    core.run(steps=10)
    for (_, product) in core.products
        output[product.name][step] = product.get()[1]
    end 
    output["z"][step]=environment.__getitem__("z")[1]
end 

using Plots
plots = []
for (_, product) in core.products
    append!(plots, [plot(output[product.name], output["z"], ylabel="z [m]", xlabel=product.unit, title=product.description)])
end
plot(plots..., layout=(1,3))
savefig("plot.svg")

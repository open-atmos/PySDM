% imports
PySDM = py.importlib.import_module('PySDM');
PySDM_environments = py.importlib.import_module('PySDM.environments');
PySDM_products = py.importlib.import_module('PySDM.products');
PySDM_initialisation = py.importlib.import_module('PySDM.initialisation');
PySDM_initialisation_spectra = py.importlib.import_module('PySDM.initialisation.spectra');
PySDM_initialisation_spectral_sampling = py.importlib.import_module('PySDM.initialisation.spectral_sampling');
PySDM_physics = py.importlib.import_module('PySDM.physics');
PySDM_dynamics = py.importlib.import_module('PySDM.dynamics');
PySDM_backends = py.importlib.import_module('PySDM.backends');

% parameters
si = PySDM_physics.constants.si;

n_sd = 100;
kappa = 1;
spectrum = PySDM_initialisation_spectra.Lognormal(pyargs(...
    'norm_factor', 1000 / si.milligram, ...
    'm_mode', 50 * si.nanometre, ...
    's_geom', 1.4 * si.dimensionless ...
));
environment = PySDM_environments.Parcel(pyargs( ...
    'dt', 1 * si.s, ...
    'mass_of_dry_air', 1 * si.kg, ...
    'p0', 1000 * si.hPa, ...
    'q0', 20 * si.g / si.kg, ...
    'T0', 300 * si.K, ...
    'w', 1 * si.m / si.s ...
));
size_range = py.list({10.633 * si.nanometre, 513.06 * si.nanometre});
radius_range = py.list({.5 * si.um, 25 * si.um});
steps = 100;
substeps = 10;

% PySDM components
builder = PySDM.Builder(pyargs( ...
    'backend', PySDM_backends.CPU, ...
    'n_sd', int32(n_sd) ...
));
builder.set_environment(environment);
builder.add_dynamic(PySDM_dynamics.AmbientThermodynamics())
builder.add_dynamic(PySDM_dynamics.Condensation(pyargs('kappa', kappa)))

tmp = PySDM_initialisation_spectral_sampling.Logarithmic(pyargs(...
    'spectrum', spectrum,...
    'size_range', size_range...
)).sample(pyargs('n_sd', builder.core.n_sd));
r_dry = tmp{1};
specific_concentration = tmp{2};
cell_id = py.numpy.zeros(pyargs('shape', int32(n_sd), 'dtype', py.numpy.int32));
r_wet = PySDM_initialisation.r_wet_init(r_dry, environment, cell_id, kappa);

attributes = py.dict(pyargs( ...
    'dry volume', PySDM_physics.formulae.volume(r_dry), ...
    'n', PySDM_initialisation.multiplicities.discretise_n(specific_concentration * environment.mass_of_dry_air), ...
    'volume', PySDM_physics.formulae.volume(r_wet) ...
));

products = py.list({ ...
    PySDM_products.ParticleMeanRadius(), ...
    PySDM_products.RelativeHumidity(), ...
    PySDM_products.CloudConcentration(pyargs('radius_range', radius_range)) ...
});

core = builder.build(pyargs( ...
    'attributes', attributes, ...
    'products', products ...
)); 

% Matlab table for output storage 
output_size = [steps+1, 1 + length(py.list(core.products.keys()))];
output_types = repelem({'double'}, output_size(2));
output_names = ['z', cellfun(@string, cell(py.list(core.products.keys())))];
output = table(...
    'Size', output_size, ...
    'VariableTypes', output_types, ...
    'VariableNames', output_names ...
);
for pykey = py.list(keys(core.products))
    get = py.getattr(core.products{pykey{1}}.get(), '__getitem__');
    key = string(pykey{1});
    output{1, key} = get(int32(0));
end
get = py.getattr(environment, '__getitem__');
zget = py.getattr(get('z'), '__getitem__');
output{1, 'z'} = zget(int32(0));

% simulation
for i=2:steps+1
    core.run(pyargs('steps', int32(substeps)));
    for pykey = py.list(keys(core.products))
        get = py.getattr(core.products{pykey{1}}.get(), '__getitem__');
        key = string(pykey{1});
        output{i, key} = get(int32(0));
    end
    output{i, 'z'} = zget(int32(0));
end

% plotting
i=1;
for pykey = py.list(keys(core.products))
    product = core.products{pykey{1}};
    subplot(1, width(output)-1, i);
    plot(output{:, string(pykey{1})}, output.z);
    title(string(product.description));
    xlabel(string(product.unit));
    ylabel('z [m]');
    i=i+1;
end

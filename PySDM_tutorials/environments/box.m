% imports
PySDM = py.importlib.import_module('PySDM');
PySDM_environments = py.importlib.import_module('PySDM.environments');
PySDM_products = py.importlib.import_module('PySDM.products');
PySDM_initialisation_spectra = py.importlib.import_module('PySDM.initialisation.spectra');
PySDM_initialisation_spectral_sampling = py.importlib.import_module('PySDM.initialisation.spectral_sampling');
PySDM_physics = py.importlib.import_module('PySDM.physics');
PySDM_dynamics = py.importlib.import_module('PySDM.dynamics');
PySDM_dynamics_coalescence_kernels = py.importlib.import_module('PySDM.dynamics.coalescence.kernels');
PySDM_backends = py.importlib.import_module('PySDM.backends');

% parameters
si = PySDM_physics.constants.si;

n_sd = 2 ^ 17;

initial_spectrum = PySDM_initialisation_spectra.Exponential(pyargs(...
    'norm_factor', 8.39e12, ...
    'scale', 1.19e5 * si.um ^ 3 ...
));
sampling_range = py.list({ ...
    PySDM_physics.formulae.volume(pyargs('radius', 10 * si.um)), ...
    PySDM_physics.formulae.volume(pyargs('radius', 100 * si.um)) ...
});

builder = PySDM.Builder(pyargs('n_sd', int32(n_sd), 'backend', PySDM_backends.CPU));
environment = PySDM_environments.Box(pyargs('dt', 1 * si.s, 'dv', 1e6 * si.m ^ 3));
builder.set_environment(environment);
attributes = environment.init_attributes(pyargs( ...
    'spectral_discretisation', ...
    PySDM_initialisation_spectral_sampling.ConstantMultiplicity(initial_spectrum)...
));
builder.add_dynamic(PySDM_dynamics.Coalescence(pyargs( ...
  'kernel', PySDM_dynamics_coalescence_kernels.Golovin(1.5e3 / si.s)) ...
));
products = py.list({ PySDM_products.ParticlesVolumeSpectrum() });
core = builder.build(attributes, products);

radius_bins_edges = logspace(log10(10 * si.um), log10(5e3 * si.um), 32);

for step = 0:1200:3600
    core.run(int32(step - core.n_steps))
    x = radius_bins_edges / si.um;
    y = core.products{"dv/dlnr"}.get(py.numpy.array(radius_bins_edges)) * PySDM_physics.constants.rho_w / si.g;
    stairs(...
        x(1:end-1), ... 
        double(py.array.array('d',py.numpy.nditer(y))), ...
        'DisplayName', sprintf("t = %d s", step) ...
    );
    hold on
end
hold off
set(gca,'XScale','log');
xlabel('particle radius [µm]')
ylabel("dm/dlnr [g/m^3/(unit dr/r)]")
legend()

import os
import numpy as np
import json

from matplotlib import pyplot as plt
from PySDM_examples.Ware_et_al_2025.example import run,Settings,SpectrumPlotter
from PySDM.initialisation.sampling.spectral_sampling import AlphaSampling
from PySDM.backends import CPU


def main(plot: bool = True, save: str = None):
    backend = CPU()
    n_sd = 16
    dt = 20#, "adaptive"]
    alphas = np.linspace(0, 1, 6)
    regular = []
    adaptive = []
    iters_without_warmup = 30
    base_error = None

    deficits_all = []
    errors_all = []
    errors_std_all = []
    errors_adaptive_all = []
    errors_adaptive_std_all = []
    


    for k,alpha in enumerate(alphas):

        plotter = SpectrumPlotter(Settings(), legend=False)
        plotter.smooth = False

        errors = []
        outputs = []
        deficits = []
        exec_times = []
        one_for_warmup = 1
        for it in range(iters_without_warmup + one_for_warmup):
            settings = Settings()
            backend.formulae.seed = it

            settings.n_sd = 2**n_sd
            settings.dt = dt 
            settings.adaptive = False
            settings.sampling = AlphaSampling(settings.spectrum,alpha=alpha,convert_to="radius")

            states, exec_time, deficit = run(settings,backend)
            deficit *= settings.dv
            print(f"{dt=}, {n_sd=}, {exec_time=}, {it=}")
            if it != 0:
                exec_times.append(exec_time)
                outputs.append(states)
                deficits.append(deficit)

                for step, vals in states.items():
                    error = plotter.plot(vals, step * settings.dt)
                errors.append(error*1e-3) #grams to kg

        mean_output = {}

        for key in outputs[0].keys():
            mean_output[key] = sum((output[key] for output in outputs)) / len(
                outputs
            )

        errors_all.append(np.mean(errors))
        errors_std_all.append(np.std(errors))
        deficits_all.append(deficits)

    backend = CPU()

    #adaptive
    for k,alpha in enumerate(alphas):

        outputs = []

        errors = []
        one_for_warmup = 1
        for it in range(iters_without_warmup + one_for_warmup):
            settings = Settings()
            backend.formulae.seed = it

            settings.n_sd = 2**n_sd
            settings.dt = dt 
            settings.adaptive = True
            settings.sampling = AlphaSampling(settings.spectrum,alpha=alpha)

            states, exec_time, deficit = run(settings,backend=backend)
            deficit *= settings.dt*settings.dv*settings.rho
            print(f"{dt=}, {n_sd=}, {exec_time=}, {it=}")
            if it != 0:
                outputs.append(states)

                for step, vals in states.items():
                    error = plotter.plot(vals, step * settings.dt)
                errors.append(error*1e-3) #grams to kg

        mean_deficit = sum(deficits) / len(deficits)

        errors_adaptive_all.append(np.mean(errors))
        errors_adaptive_std_all.append(np.std(errors))


    return n_sd, alphas,deficits_all,errors_all,errors_std_all,errors_adaptive_all,errors_adaptive_std_all

n_sd,alphas,deficits_all,errors_all,errors_std_all,errors_adaptive_all,errors_adaptive_std_all = main(plot=False, save=".")
    

# %%
plt.figure(figsize=(4, 4))
import matplotlib.pyplot as plt
plt.errorbar(alphas,np.array(errors_all)*1e3,yerr=np.array(errors_std_all)*1e3,
             marker='o', color='red', ls='--',
            lw=0.5, 
            capsize=5, capthick=1, ecolor='black',
             label="Fixed")
plt.errorbar(alphas,np.array(errors_adaptive_all)*1e3,yerr= np.array(errors_adaptive_std_all)*1e3,
             marker='s', color='blue', ls='--',
            lw=0.5, 
             capsize=5, capthick=1, ecolor='black',
             label="Adaptive")
plt.legend()
plt.tight_layout(pad=2.0)
plt.title(f"$N_s=2^{{{n_sd}}}$, $\Delta t:20$ s")
plt.ylabel(f"RMSE at t=3600s\n[g/m$^3$/unit ln(r)]")
plt.xlabel(r'$\alpha$' +f'\n0 = constant-multiplicity, 1 = uniform-in-$r$')
plt.savefig(f"alpha_sampling_comparison{n_sd}.pdf", bbox_inches='tight')

  # %%

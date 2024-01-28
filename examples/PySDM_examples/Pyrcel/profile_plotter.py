import numpy as np
from matplotlib import pyplot
from open_atmos_jupyter_utils import show_plot

from PySDM.physics.constants import si


class ProfilePlotter:
    def __init__(self, settings, legend=True, log_base=10):
        self.settings = settings
        self.format = "pdf"
        self.legend = legend
        self.log_base = log_base
        self.ax = pyplot
        self.fig = pyplot

    def show(self):
        pyplot.tight_layout()
        show_plot()

    def save(self, file):
        # self.finish()
        pyplot.savefig(file, format=self.format)

    def plot(self, output):
        self.plot_data(self.settings, output)

    def plot_data(self, settings, output):
        _, axs = pyplot.subplots(1, 2, sharey=True, figsize=(10, 5))
        axS = axs[0]
        axS.plot(
            np.asarray(output["products"]["S_max"]) - 100,
            output["products"]["z"],
            color="black",
        )
        axS.set_ylabel("Displacement [m]")
        axS.set_xlabel("Supersaturation [%]")
        axS.set_xlim(0, 0.7)
        axS.set_ylim(0, 250)
        axS.text(0.3, 52, f"max S = {np.nanmax(output['products']['S_max'])-100:.2f}%")
        axS.grid()

        axT = axS.twiny()
        axT.xaxis.label.set_color("red")
        axT.tick_params(axis="x", colors="red")
        axT.plot(output["products"]["T"], output["products"]["z"], color="red")
        rng = (272, 274)
        axT.set_xlim(*rng)
        axT.set_xticks(np.linspace(*rng, num=5))
        axT.set_xlabel("Temperature [K]")

        axR = axs[1]
        axR.set_xscale("log")
        axR.set_xlim(1e-2, 1e2)
        for drop_id, volume in enumerate(output["attributes"]["volume"]):
            axR.plot(
                settings.formulae.trivia.radius(volume=np.asarray(volume)) / si.um,
                output["products"]["z"],
                color="magenta" if drop_id < settings.n_sd_per_mode[0] else "blue",
                label=(
                    "mode 1"
                    if drop_id == 0
                    else "mode 2" if drop_id == settings.n_sd_per_mode[0] else ""
                ),
            )
        axR.legend(loc="upper right")
        axR.set_xlabel("Droplet radius [μm]")

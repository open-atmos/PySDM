import matplotlib
import numpy as np
from matplotlib import pyplot
from open_atmos_jupyter_utils import show_plot
from packaging import version
from PySDM_examples.Shima_et_al_2009.error_measure import error_measure

from PySDM.physics.constants import si

_matplotlib_version_3_3_3 = version.parse("3.3.0")
_matplotlib_version_actual = version.parse(matplotlib.__version__)


class SpectrumColors:
    def __init__(self, begining="#2cbdfe", end="#b317b1"):
        self.b = begining
        self.e = end

    def __call__(self, value: float):
        bR, bG, bB = int(self.b[1:3], 16), int(self.b[3:5], 16), int(self.b[5:7], 16)
        eR, eG, eB = int(self.e[1:3], 16), int(self.e[3:5], 16), int(self.e[5:7], 16)
        R = bR + int((eR - bR) * value)
        G = bG + int((eG - bG) * value)
        B = bB + int((eB - bB) * value)
        result = f"#{hex(R)[2:4]}{hex(G)[2:4]}{hex(B)[2:4]}"
        return result


class SpectrumPlotter:
    def __init__(self, settings, title=None, grid=True, legend=True, log_base=10):
        self.settings = settings
        self.format = "pdf"
        self.colors = SpectrumColors()
        self.smooth = False
        self.smooth_scope = 2
        self.legend = legend
        self.grid = grid
        self.title = title
        self.xlabel = "particle radius [µm]"
        self.ylabel = "dm/dlnr [g/m^3/(unit dr/r)]"
        self.log_base = log_base
        self.ax = pyplot
        self.fig = pyplot
        self.finished = False

    def finish(self):
        if self.finished:
            return
        self.finished = True
        if self.grid:
            self.ax.grid()

        base_arg = {
            "base"
            + (
                "x" if _matplotlib_version_actual < _matplotlib_version_3_3_3 else ""
            ): self.log_base
        }
        if self.title is not None:
            try:
                self.ax.title(self.title)
            except TypeError:
                self.ax.set_title(self.title)
        try:
            self.ax.xscale("log", **base_arg)
            self.ax.xlabel(self.xlabel)
            self.ax.ylabel(self.ylabel)
        except AttributeError:
            self.ax.set_xscale("log", **base_arg)
            self.ax.set_xlabel(self.xlabel)
            self.ax.set_ylabel(self.ylabel)
        if self.legend:
            self.ax.legend()

    def show(self):
        self.finish()
        pyplot.tight_layout()
        show_plot()

    def save(self, file):
        self.finish()
        pyplot.savefig(file, format=self.format)

    def plot(self, spectrum, t):
        error = self.plot_analytic_solution(self.settings, t, spectrum)
        self.plot_data(self.settings, t, spectrum)
        return error

    def plot_analytic_solution(self, settings, t, spectrum=None):
        if t == 0:
            analytic_solution = settings.spectrum.size_distribution
        else:

            def analytic_solution(x):
                return settings.norm_factor * settings.kernel.analytic_solution(
                    x=x, t=t, x_0=settings.X0, N_0=settings.n_part
                )

        volume_bins_edges = self.settings.formulae.trivia.volume(
            settings.radius_bins_edges
        )
        dm = np.diff(volume_bins_edges)
        dr = np.diff(settings.radius_bins_edges)

        pdf_m_x = volume_bins_edges[:-1] + dm / 2
        pdf_m_y = analytic_solution(pdf_m_x)

        pdf_r_x = settings.radius_bins_edges[:-1] + dr / 2
        pdf_r_y = pdf_m_y * dm / dr * pdf_r_x

        x = pdf_r_x * si.metres / si.micrometres
        y_true = (
            pdf_r_y
            * self.settings.formulae.trivia.volume(radius=pdf_r_x)
            * settings.rho
            / settings.dv
            * si.kilograms
            / si.grams
        )

        self.ax.plot(x, y_true, color="black")

        if spectrum is not None:
            y = spectrum * si.kilograms / si.grams
            error = error_measure(y, y_true, x)
            self.title = f"error measure: {error:.2f}"  # TODO #327 relative error
            return error
        return None

    def plot_data(self, settings, t, spectrum):
        if self.smooth:
            scope = self.smooth_scope
            if t != 0:
                new = np.copy(spectrum)
                for _ in range(2):
                    for i in range(scope, len(spectrum) - scope):
                        new[i] = np.mean(spectrum[i - scope : i + scope + 1])
                    scope = 1
                    for i in range(scope, len(spectrum) - scope):
                        spectrum[i] = np.mean(new[i - scope : i + scope + 1])

            x = settings.radius_bins_edges[:-scope]
            dx = np.diff(x)
            self.ax.plot(
                (x[:-1] + dx / 2) * si.metres / si.micrometres,
                spectrum[:-scope] * si.kilograms / si.grams,
                label=f"t = {t}s",
                color=self.colors(
                    t / (self.settings.output_steps[-1] * self.settings.dt)
                ),
            )
        else:
            self.ax.step(
                settings.radius_bins_edges[:-1] * si.metres / si.micrometres,
                spectrum * si.kilograms / si.grams,
                where="post",
                label=f"t = {t}s",
                color=self.colors(
                    t / (self.settings.output_steps[-1] * self.settings.dt)
                ),
            )

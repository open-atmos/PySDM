"""shared plot functions for homogeneous freezing notebooks"""

from matplotlib import pyplot, ticker
import numpy as np
import seaborn as sns

from PySDM import Formulae

formulae = Formulae(
    particle_shape_and_density="MixedPhaseSpheres",
)

# general plot settings
ax_title_size = 18
ax_lab_fsize = 15
tick_fsize = 15
T_frz_bins = np.linspace(-40, -34, num=60, endpoint=True)
T_frz_bins_kelvin = np.linspace(230, 240, num=100, endpoint=True)


def cumulative_histogram(data, bins, reverse=False, density=True):
    # Compute regular histogram using given bins
    hist, bin_edges = np.histogram(data, bins=bins, density=False)

    # Cumulative sum
    if reverse:
        cum_hist = np.cumsum(hist[::-1])[::-1]
        cum_hist_0 = cum_hist[0]
    else:
        cum_hist = np.cumsum(hist)
        cum_hist_0 = cum_hist[-1]

    # Normalize
    if density:
        cum_hist = cum_hist / cum_hist_0

    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return cum_hist, bin_centers


def plot_thermodynamics_and_bulk(simulation, title_add="", show_conc=False):

    output = simulation["ensemble_member_outputs"][0]
    time = output["t"]
    T = np.asarray(output["T"])
    RH = np.asarray(output["RH"])
    RHi = np.asarray(output["RH_ice"])
    qc = np.asarray(output["LWC"])
    qi = np.asarray(output["IWC"])
    qv = np.asarray(output["qv"])
    qt = qc + qv + qi
    rc = np.asarray(output["rs"])
    ri = np.asarray(output["ri"])
    if show_conc:
        nc = np.asarray(output["ns"])
        ni = np.asarray(output["ni"])

    svp = Formulae(
        saturation_vapour_pressure="FlatauWalkoCotton"
    ).saturation_vapour_pressure
    a_w_ice = svp.pvs_ice(T) / svp.pvs_water(T)
    d_a_w_ice = (RHi / 100 - 1) * a_w_ice

    j_hom_rate = Formulae(
        homogeneous_ice_nucleation_rate="KoopMurray2016"
    ).homogeneous_ice_nucleation_rate
    koop_murray_2016 = j_hom_rate.j_hom(T, d_a_w_ice)
    j_hom_rate = Formulae(
        homogeneous_ice_nucleation_rate="Koop_Correction"
    ).homogeneous_ice_nucleation_rate
    spichtinger_2023 = j_hom_rate.j_hom(T, d_a_w_ice)

    fig, axs = pyplot.subplots(
        2, 2, figsize=(10, 10), sharex=False, constrained_layout=True
    )

    title = (
        "Freezing: "
        + simulation["settings"]["hom_freezing"]
        + "  n_sd: "
        + str(simulation["settings"]["n_sd"])
        + "  w: "
        + str(simulation["settings"]["w_updraft"])
        + r" $\mathrm{m \, s^{-1}}$"
        + title_add
    )
    fig.suptitle(title, fontsize=16)

    # Temperture profile
    ax = axs[0, 0]
    ax.plot(time, formulae.trivia.K2C(T), color="black", linestyle="-", label="T")
    ax.set_xlabel("time [s]", fontsize=ax_lab_fsize)
    ax.set_ylabel("temperature [°C]", fontsize=ax_lab_fsize)
    ax.legend(loc="upper left", fontsize=ax_lab_fsize)
    ax.tick_params(labelsize=tick_fsize)

    twin = ax.twinx()
    twin.plot(time, RH, color="red", linestyle="-", label=r"$S_{w}$")
    twin.plot(time, RHi, color="blue", linestyle="-", label=r"$S_{i}$")
    twin.set_ylabel("relative humidity [%]", fontsize=ax_lab_fsize)
    twin.legend(loc="center right", fontsize=ax_lab_fsize)
    twin.tick_params(labelsize=tick_fsize)
    twin.grid(visible=True)

    # Water activity difference profile
    lin_s_SP2023 = "--"
    lin_s_KM2016 = "-"
    if simulation["settings"]["hom_freezing"] == "Spichtinger2023":
        lin_s_SP2023 = "-"
        lin_s_KM2016 = "--"
    ax = axs[0, 1]
    ax.plot(
        time,
        koop_murray_2016,
        color="blue",
        linestyle=lin_s_KM2016,
        label="KoopMurray2016",
    )
    ax.plot(
        time,
        spichtinger_2023,
        color="red",
        linestyle=lin_s_SP2023,
        label="Spichtinger2023",
    )
    ax.set_xlabel("time [s]", fontsize=ax_lab_fsize)
    ax.set_ylabel(
        r"nucleation rate [$\mathrm{m^{-3} \, s^{-1}}$]", fontsize=ax_lab_fsize
    )
    ax.legend(loc="upper left", fontsize=ax_lab_fsize)
    ax.set_ylim(1e-30, 1e30)
    ax.set_yscale("log")
    ax.tick_params(labelsize=tick_fsize)
    ax.grid(visible=True)
    twin = ax.twinx()
    twin.plot(time, d_a_w_ice, color="gray", linestyle="-", label=r"$\Delta a_{w}$")
    twin.set_ylim(0.2, 0.35)
    twin.set_ylabel("water activity difference", fontsize=ax_lab_fsize)
    twin.tick_params(labelsize=tick_fsize)
    twin.legend(loc="lower right", fontsize=ax_lab_fsize)

    # Mass content and number concentration
    ax = axs[1, 0]
    ax.plot(time, qc, color="red", linestyle="-", label="water")
    ax.plot(time, qi, color="blue", linestyle="-", label="ice")
    ax.plot(time, qv, color="black", linestyle="-", label="vapor")
    ax.plot(time, qt, color="black", linestyle="dotted", label="total")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e-2)
    ax.set_xlabel("time [s]", fontsize=ax_lab_fsize)
    ax.set_ylabel(r"mass content [$\mathrm{kg \, kg^{-1}}$]", fontsize=ax_lab_fsize)
    ax.legend(fontsize=ax_lab_fsize)
    ax.tick_params(labelsize=tick_fsize)
    ax.grid(visible=True)

    if show_conc:
        twin = ax.twinx()
        twin.plot(time, nc, color="red", linestyle="--", label="water")
        twin.plot(time, ni, color="blue", linestyle="--", label="ice")
        twin.set_yscale("log")
        twin.set_xlabel("time [s]", fontsize=ax_lab_fsize)
        twin.set_ylabel(
            r"number concentration [$\mathrm{kg^{-1}}$]", fontsize=ax_lab_fsize
        )
        twin.tick_params(labelsize=tick_fsize)

    # Mean radius
    ax = axs[1, 1]
    ax.plot(time, rc * 1e6, color="red", linestyle="-", label="water")
    ax.plot(time, ri * 1e6, color="blue", linestyle="-", label="ice")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e2)
    ax.set_xlabel("time [s]", fontsize=ax_lab_fsize)
    ax.set_ylabel("mean radius [µm]", fontsize=ax_lab_fsize)
    ax.legend(fontsize=ax_lab_fsize)
    ax.tick_params(labelsize=tick_fsize)
    ax.grid(visible=True)


def plot_freezing_temperatures_histogram(ax, simulation):

    number_of_ensemble_runs = simulation["settings"]["number_of_ensemble_runs"]

    for i in range(number_of_ensemble_runs):
        output = simulation["ensemble_member_outputs"][i]
        T_frz = np.asarray(output["T_frz"])

        title = "Nucleation rate=" + simulation["settings"]["hom_freezing"]

        # Freezing temperatures
        ax.hist(
            T_frz,
            bins=T_frz_bins_kelvin,
            density=True,
            cumulative=-1,
            alpha=1.0,
            histtype="step",
            linewidth=1.5,
        )

        ax.set_xlim(left=234, right=239)
        ax.axvline(x=235, color="k", linestyle="--")
        ax.set_title(title, fontsize=ax_lab_fsize)
        ax.set_xlabel("freezing temperature [K]", fontsize=ax_lab_fsize)
        ax.set_ylabel("frozen fraction", fontsize=ax_lab_fsize)
        ax.tick_params(labelsize=tick_fsize)

    return ax


def plot_freezing_temperatures_histogram_allinone(
    ax, simulations, title, lloc="upper right"
):

    colors = ["black", "blue", "red"]

    for k, simulation in enumerate(simulations):

        number_of_ensemble_runs = simulation["settings"]["number_of_ensemble_runs"]
        n_sd = simulation["settings"]["n_sd"]
        histogram_list = np.zeros((number_of_ensemble_runs, len(T_frz_bins_kelvin) - 1))
        for i in range(number_of_ensemble_runs):
            output = simulation["ensemble_member_outputs"][i]
            T_frz = np.asarray(output["T_frz"])

            hist, T_frz_bins_center = cumulative_histogram(
                T_frz, T_frz_bins_kelvin, reverse=True
            )
            histogram_list[i, :] = hist

        max_line = np.max(histogram_list, axis=0)
        mean_line = np.mean(histogram_list, axis=0)
        min_line = np.min(histogram_list, axis=0)

        color = colors[k]
        ax.plot(
            T_frz_bins_center,
            mean_line,
            color=color,
            label=r"$N_{sd}$: " + f"{int(n_sd):5.0f}",
        )
        ax.fill_between(T_frz_bins_center, min_line, max_line, color=color, alpha=0.2)

    ax.set_xlim(left=234.5, right=239)
    ax.axvline(x=235, color="k", linestyle="--")
    ax.set_title(title, fontsize=ax_lab_fsize)
    ax.set_xlabel("freezing temperature [K]", fontsize=ax_lab_fsize)
    ax.set_ylabel("frozen fraction", fontsize=ax_lab_fsize)
    ax.tick_params(labelsize=tick_fsize)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.legend(loc=lloc, fontsize=ax_lab_fsize)

    return ax


def plot_freezing_temperatures_2d_histogram_seaborn(
    ensemble_simulations, hom_freezing_type, calc_pairwise_distance=False, title=""
):

    sns.set_theme(style="ticks")

    ens_variable_name = ensemble_simulations["ens_variable_name"]
    simulations = ensemble_simulations[hom_freezing_type]

    ens_variable = []
    T_frz_hist = []

    for simulation in simulations:
        ens_variable_value = simulation["settings"][ens_variable_name]
        n_realisations = simulation["settings"]["number_of_ensemble_runs"]
        n_sd = simulation["settings"]["n_sd"]

        T_frz_realisations = np.zeros((n_realisations, n_sd))

        for i in range(n_realisations):
            output = simulation["ensemble_member_outputs"][i]
            T_frz_realisations[i, :] = np.asarray(output["T_frz"])

        if calc_pairwise_distance:
            pass
        else:
            T_frz = np.mean(T_frz_realisations, axis=0)

        T_frz_hist.extend(T_frz)
        ens_variable.extend(np.full_like(T_frz, ens_variable_value))

    y_label = ""
    if ens_variable_name == "w_updraft":
        y_label = r"vertical updraft [$\mathrm{m \, s^{-1}}$]"
    elif ens_variable_name == "n_ccn":
        y_label = r"ccn concentration [$\mathrm{m^{-3}}$]"
    elif ens_variable_name == "maximum_radius":
        ens_variable = ens_variable * 1e6
        y_label = "(maximum) radius [µm]"

    xlim = (233, 241)
    h = sns.JointGrid(
        x=T_frz_hist,
        y=ens_variable,
        xlim=xlim,
    )
    h.ax_joint.set(yscale="log")
    ax = h.figure.add_axes([0.75, 0.55, 0.02, 0.2])

    h.plot_joint(
        sns.histplot,
        stat="probability",
        binwidth=0.25,
        discrete=(False, False),
        pmax=0.8,
        cbar=True,
        cbar_ax=ax,
    )

    h.plot_marginals(
        sns.histplot,
        element="step",
    )
    h.set_axis_labels("freezing temperature [K]", y_label, fontsize=ax_lab_fsize)
    h.ax_joint.set_title(
        title,
        pad=70,
        fontsize=ax_title_size,
    )
    h.ax_marg_y.remove()

    return ax


def plot_ensemble_bulk(
    ax, ensemble_simulations, var_name, title_add=""
):  # pylint: disable=too-many-nested-blocks

    for ensemble_simulation in ensemble_simulations:

        ens_var = ensemble_simulation["ens_variable"]
        ens_var_name = ensemble_simulation["ens_variable_name"]
        hom_freezing_types = ensemble_simulation["hom_freezing_types"]
        hom_freezing_labels = ["KM16", "SP23"]
        len_ens_var = len(ens_var)

        for k, hom_freezing_type in enumerate(hom_freezing_types):
            simulations = ensemble_simulation[hom_freezing_type]
            var = np.zeros(len_ens_var)
            for i in range(len_ens_var):
                for simulation in simulations:
                    if simulation["settings"][ens_var_name] == ens_var[i]:
                        output = simulation["ensemble_member_outputs"][0]
                        if var_name == "freezing_fraction":
                            ni = np.asarray(output["ni"])[-1]
                            nc = np.asarray(output["ns"])[0]
                            var[i] = (1 - (nc - ni) / nc) * 100
                        else:
                            var[i] = np.asarray(output[var_name])[-1]

            ax.plot(var, ens_var, "-o", label=hom_freezing_labels[k])

    title, x_label, y_label, ens_label = "", "", "", ""
    if var_name == "ni":
        ax.set_xscale("log")
        x_label = r"$n_{i} \, [\mathrm{kg^{-1}}$]"
        title = "ice number concentrations"
        ax.set_xlim(1e6, 1e10)

    if var_name == "IWC":
        ax.set_xscale("log")
        x_label = r"mass content [$\mathrm{kg \, kg^{-1}}$]"
        title = "ice mass content"

    if var_name == "freezing_fraction":
        title = "frozen fraction"
        x_label = r"$n_{frz} \, [\mathrm{\%}$]"
        ax.set_xlim(0, 20)

    if ens_var_name == "n_ccn":
        ax.set_yscale("log")
        y_label = r"$n_{ccn} \, [\mathrm{m^{-3}}$]"
        ens_label = r"$n_{ccn}$ ensemble"

    if ens_var_name == "w_updraft":
        ax.set_yscale("log")
        y_label = r"w [$\mathrm{m \, s^{-1}}$]"
        ens_label = "w ensemble"

    ax.set_title(title_add + " " + title + " for " + ens_label, fontsize=ax_lab_fsize)
    ax.set_xlabel(x_label, fontsize=ax_lab_fsize)
    ax.set_ylabel(y_label, fontsize=ax_lab_fsize)
    ax.legend(fontsize=ax_lab_fsize)

    return ax

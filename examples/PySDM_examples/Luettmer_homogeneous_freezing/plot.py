"""shared plot functions for homogeneous freezing notebooks"""

from matplotlib import pyplot, ticker
import numpy as np
import seaborn as sns
from cycler import cycler
from PySDM import Formulae

formulae = Formulae(particle_shape_and_density="MixedPhaseSpheres")

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


def plot_thermodynamics_and_bulk(
    simulation,
    title_add=None,
    show_conc=False,
    show_jhom=True,
    show_tf=True,
    t_lim=None,
):
    if title_add is None:
        title_add = ["", "", "", ""]
    plot_daw = False
    output = simulation["ensemble_member_outputs"][0]
    time = output["t"]
    T = np.asarray(output["T"])
    RH = np.asarray(output["RH"]) / 100
    RHi = np.asarray(output["RH_ice"]) / 100
    qc = np.asarray(output["LWC"])
    qi = np.asarray(output["IWC"])
    qv = np.asarray(output["qv"])
    qt = qc + qv + qi
    T_frz = np.asarray(output["T_frz"])
    if show_conc:
        nc = np.asarray(output["ns"])
        ni = np.asarray(output["ni"])

    if t_lim is None:
        t_lim = np.amax(time)

    first_ice_idx = np.where(qi > 1e-7)[0][0]
    first_ice_time = time[first_ice_idx]
    first_T_frz = T[first_ice_idx]

    if not show_tf:
        rc = np.asarray(output["rs"])
        ri = np.asarray(output["ri"])

    if show_jhom:
        svp = Formulae(
            saturation_vapour_pressure="FlatauWalkoCotton"
        ).saturation_vapour_pressure
        a_w_ice = svp.pvs_ice(T) / svp.pvs_water(T)
        d_a_w_ice = (RHi - 1) * a_w_ice

        j_hom_rate = Formulae(
            homogeneous_ice_nucleation_rate="KoopMurray2016"
        ).homogeneous_ice_nucleation_rate
        koop_murray_2016 = j_hom_rate.j_hom(T, d_a_w_ice)
        j_hom_rate = Formulae(
            homogeneous_ice_nucleation_rate="Koop_Correction"
        ).homogeneous_ice_nucleation_rate
        spichtinger_2023 = j_hom_rate.j_hom(T, d_a_w_ice)
        abs_diff_j_hom = (koop_murray_2016 - spichtinger_2023) / koop_murray_2016
    else:
        radius = np.asarray(output["radius"])
        multiplicity = np.asarray(output["multiplicity"])

    _, axs = pyplot.subplots(
        1, 4, figsize=(20, 5), sharex=False, constrained_layout=True
    )

    # Temperture profile
    iax = 0
    ax = axs[iax]

    ax.plot(time, RH, color="red", linestyle="-", label=r"$S_{w}$")
    ax.plot(time, RHi, color="blue", linestyle="-", label=r"$S_{i}$")
    ax.set_ylabel("saturation ratio", fontsize=ax_lab_fsize)
    ax.legend(loc="center left", fontsize=ax_lab_fsize)
    ax.set_xlim(time[0], t_lim)
    ax.tick_params(labelsize=tick_fsize)
    ax.set_title(title_add[iax] + r"ambient thermodynamics", fontsize=ax_lab_fsize)
    ax.grid(visible=True)
    ax.axvline(x=first_ice_time, color="black", linestyle="--")

    twin = ax.twinx()
    twin.plot(time, T, color="black", linestyle="-", label="T")
    twin.set_xlabel("time [s]", fontsize=ax_lab_fsize)
    twin.set_ylabel("temperature [K]", fontsize=ax_lab_fsize)
    twin.legend(loc="upper left", fontsize=ax_lab_fsize)
    twin.tick_params(labelsize=tick_fsize)

    # Mass content and number concentration
    iax = 1
    ax = axs[iax]
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
    ax.set_xlim(time[0], t_lim)
    ax.grid(visible=True)
    ax.axvline(x=first_ice_time, color="black", linestyle="--")
    ax.set_title(title_add[iax] + r"bulk quantities", fontsize=ax_lab_fsize)
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

    iax = 2
    ax = axs[iax]
    if show_tf:
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
        ax.set_xlabel("freezing temperature [K]", fontsize=ax_lab_fsize)
        ax.set_ylabel("frozen fraction", fontsize=ax_lab_fsize)
        ax.tick_params(labelsize=tick_fsize)
        ax.grid(visible=True)
        ax.axvline(x=first_T_frz, color="black", linestyle="--")
        ax.set_title(title_add[iax] + r"$T_{frz}$ histogram", fontsize=ax_lab_fsize)
    else:
        ax.plot(time, rc * 1e6, color="red", linestyle="-", label="water")
        ax.plot(time, ri * 1e6, color="blue", linestyle="-", label="ice")
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 1e2)
        ax.set_xlabel("time [s]", fontsize=ax_lab_fsize)
        ax.set_ylabel("mean radius [µm]", fontsize=ax_lab_fsize)
        ax.legend(fontsize=ax_lab_fsize)
        ax.set_xlim(time[0], t_lim)
        ax.tick_params(labelsize=tick_fsize)
        ax.grid(visible=True)
        ax.axvline(x=first_ice_time, color="black", linestyle="--")
        ax.set_title(title_add[iax] + r" mean radius", fontsize=ax_lab_fsize)

    iax = 3
    # Water activity difference profile
    ax = axs[iax]
    if show_jhom:
        lin_s_SP2023 = "--"
        lin_s_KM2016 = "-"
        if simulation["settings"]["hom_freezing"] == "Spichtinger2023":
            lin_s_SP2023 = "-"
            lin_s_KM2016 = "--"

        ax.plot(
            time,
            koop_murray_2016,
            color="blue",
            linestyle=lin_s_KM2016,
            label="KM16",
        )
        ax.plot(
            time,
            spichtinger_2023,
            color="red",
            linestyle=lin_s_SP2023,
            label="SP23",
        )

        ax.set_ylabel(
            r"nucleation rate [$\mathrm{m^{-3} \, s^{-1}}$]", fontsize=ax_lab_fsize
        )
        ax.set_ylim(1e-30, 1e30)
        ax.set_title(title_add[iax] + r"nucleation rates", fontsize=ax_lab_fsize)
        ax.legend(loc="upper left", fontsize=ax_lab_fsize)
        ax.set_yscale("log")
        ax.set_xlim(time[0], t_lim)
        ax.axvline(x=first_ice_time, color="black", linestyle="--")
        ax.set_xlabel("time [s]", fontsize=ax_lab_fsize)
        if plot_daw:
            twin = ax.twinx()
            twin.plot(
                time, d_a_w_ice, color="gray", linestyle="-", label=r"$\Delta a_{w}$"
            )
            twin.set_ylim(0.2, 0.35)
            twin.set_ylabel("water activity difference", fontsize=ax_lab_fsize)
            twin.tick_params(labelsize=tick_fsize)
            twin.legend(loc="lower left", fontsize=ax_lab_fsize)
        else:
            twin = ax.twinx()
            twin.plot(
                time, abs_diff_j_hom, label=r"$\Delta J_{\mathrm{hom}}$", color="gray"
            )
            twin.set_ylim(-10, 10)
            twin.set_ylabel("relative error", fontsize=ax_lab_fsize)
            twin.tick_params(labelsize=tick_fsize)
            twin.legend(loc="lower left", fontsize=ax_lab_fsize)
    else:
        ax.scatter(radius * 1e6, multiplicity)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 5e-0)
        ax.set_xlabel("initial radius [µm]", fontsize=ax_lab_fsize)
        # ax.set_xlim(left=234, right=239)
        ax.set_ylabel("multiplicity", fontsize=ax_lab_fsize)
        ax.set_title(title_add[iax] + r"CCN size distribution", fontsize=ax_lab_fsize)

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
    ax.set_title(title, fontsize=ax_lab_fsize)
    ax.set_xlabel("freezing temperature [K]", fontsize=ax_lab_fsize)
    ax.set_ylabel("frozen fraction", fontsize=ax_lab_fsize)
    ax.tick_params(labelsize=tick_fsize)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    ax.grid(visible=True)
    ax.legend(loc=lloc, fontsize=ax_lab_fsize)

    return ax


def plot_freezing_temperatures_2d_histogram_seaborn(
    ensemble_simulations,
    hom_freezing_type,
    title="",
    height=4,
    width=5,
):

    sns.set_theme(style="ticks")

    second_axis = True
    y_log = True

    ens_variable_name = ensemble_simulations["ens_variable_name"]
    simulations = ensemble_simulations[hom_freezing_type]

    ens_variable = []
    ens_variable_sec = []
    T_frz_hist = []

    for simulation in simulations:
        if ens_variable_name == "sig":
            ens_variable_value = simulation["settings"]["sigma_droplet_distribution"]
        else:
            ens_variable_value = simulation["settings"][ens_variable_name]

        output = simulation["ensemble_member_outputs"][0]
        qi = np.asarray(output["IWC"])
        T_frz = np.asarray(output["T_frz"])
        first_ice_idx = np.where(qi > 1e-7)[0][0]

        if ens_variable_name == "n_ccn":
            rs = np.asarray(output["rs"])
            ens_variable_sec_value = rs[first_ice_idx - 1]
        else:
            time = np.asarray(output["t"])
            T = np.asarray(output["T"])
            ens_variable_sec_value = abs(np.mean(np.diff(T) / np.diff(time)))

        T_frz_hist.extend(T_frz)
        ens_variable.extend(np.full_like(T_frz, ens_variable_value))
        ens_variable_sec.append(ens_variable_sec_value)

    ens_variable = np.array(ens_variable)
    ens_variable_sec = np.array(ens_variable_sec)

    y_label, y_label_sec = "", ""
    if ens_variable_name == "w_updraft":
        y_label = r"vertical updraft [$\mathrm{m \, s^{-1}}$]"
        y_label_sec = r"cooling rate [$\mathrm{K \, s^{-1}}$]"
        binwidth = 0.25
    elif ens_variable_name == "n_ccn":
        y_label = r"ccn concentration [$\mathrm{cm^{-3}}$]"
        y_label_sec = r"radius [$\mathrm{\mu m}$]"
        ens_variable = ens_variable / 1.0e6
        ens_variable_sec = ens_variable_sec * 1.0e6
        binwidth = 0.25
    elif ens_variable_name == "sig":
        y_label = r"$\sigma$"
        second_axis = False
        y_log = False
        binwidth = 0.1

    ens_variable_label = np.unique(np.sort(ens_variable))

    xlim = (231.5, 240)
    h = sns.JointGrid(
        x=T_frz_hist,
        y=ens_variable,
        xlim=xlim,
        # ylim=(np.min(ens_variable), np.max(ens_variable)),
    )
    if y_log:
        h.ax_joint.set(yscale="log")
    ax = h.figure.add_axes([0.15, 0.1, 0.02, 0.2])

    h.plot_joint(
        sns.histplot,
        stat="probability",
        binwidth=binwidth,
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
        pad=50,
        fontsize=ax_title_size,
    )
    h.ax_joint.tick_params(labelsize=tick_fsize)
    h.ax_joint.grid(True, axis="x", which="both", linestyle="-", linewidth=0.6)
    h.ax_marg_x.tick_params(labelsize=tick_fsize)
    h.ax_marg_y.tick_params(labelsize=tick_fsize)

    h.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(1))
    h.ax_marg_y.remove()

    if second_axis:

        ax2 = h.ax_joint.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
        ax2.set_yticks(ens_variable_label)
        ax2.set_yticklabels([f"{v:.3f}" for v in ens_variable_sec])
        ax2.minorticks_off()
        ax2.tick_params(
            axis="y",
            which="both",
            direction="out",
            length=h.ax_joint.yaxis.get_ticklines()[0].get_markersize(),
            width=h.ax_joint.yaxis.get_ticklines()[0].get_markeredgewidth(),
            labelsize=tick_fsize,
        )
        ax2.set_ylabel(y_label_sec, fontsize=ax_lab_fsize)

    h.fig.set_size_inches(width, height)

    return ax


def plot_ensemble_bulk(
    ax, ensemble_simulations, var_name, title_add=""
):  # pylint: disable=too-many-nested-blocks

    colors = ["blue", "red", "cyan"]
    pyplot.rcParams["axes.prop_cycle"] = cycler(color=colors)

    for ensemble_simulation in ensemble_simulations:
        ens_var = np.asarray(ensemble_simulation["ens_variable"])
        ens_var_name = ensemble_simulation["ens_variable_name"]
        hom_freezing_types = ensemble_simulation["hom_freezing_types"]
        hom_freezing_labels = ensemble_simulation["hom_freezing_labels"]
        len_ens_var = len(ens_var)

        if ens_var_name == "n_ccn":
            ens_var_scale = 1.0 / 1e6
        else:
            ens_var_scale = 1.0

        if ens_var_name == "sig":
            ens_var_name = "sigma_droplet_distribution"

        for j, hom_freezing_type in enumerate(hom_freezing_types):
            simulations = ensemble_simulation[hom_freezing_type]
            number_of_ensemble_runs = simulations[0]["settings"][
                "number_of_ensemble_runs"
            ]
            var = np.zeros((len_ens_var, number_of_ensemble_runs))
            for i in range(len_ens_var):
                for simulation in simulations:
                    if simulation["settings"][ens_var_name] == ens_var[i]:
                        for h in range(number_of_ensemble_runs):
                            output = simulation["ensemble_member_outputs"][h]
                            if var_name == "freezing_fraction":
                                ni = np.asarray(output["ni"])[-1]
                                nc = np.asarray(output["ns"])[0]
                                var[i, h] = (1 - (nc - ni) / nc) * 100
                            else:
                                var[i, h] = np.asarray(output[var_name])[-1]

            if number_of_ensemble_runs > 1:
                ax.plot(
                    np.mean(var, axis=1),
                    ens_var * ens_var_scale,
                    "-o",
                    label=hom_freezing_labels[j],
                )
                # print( np.min(var,axis=1), np.max(var,axis=1),ens_var * ens_var_scale )
                ax.fill_betweenx(
                    ens_var * ens_var_scale,
                    np.min(var, axis=1),
                    np.max(var, axis=1),
                    alpha=0.2,
                )
            else:
                ax.plot(
                    var[:, 0],
                    ens_var * ens_var_scale,
                    "-o",
                    label=hom_freezing_labels[j],
                )

    title, x_label, y_label, ens_label = "", "", "", ""
    if var_name == "ni":
        ax.set_xscale("log")
        x_label = r"ice number concentration [$\mathrm{kg^{-1}}$]"
        title = r"$n_{i}$"
        ax.set_xlim(1e6, 1e10)
    elif var_name == "IWC":
        ax.set_xscale("log")
        x_label = r"mass content [$\mathrm{kg \, kg^{-1}}$]"
        title = "ice mass content"
        ax.set_xlim(1e-4, 1e-3)
    elif var_name == "freezing_fraction":
        title = r"$n_{frz}$"
        x_label = r"frozen fraction [$\mathrm{\%}$]"
        ax.set_xlim(0, 20)
    elif ens_var_name == "n_ccn":
        ax.set_yscale("log")
        y_label = r"ccn concentration [$\mathrm{cm^{-3}}$]"
        ens_label = r"$n_{ccn}$ ensemble"
    elif ens_var_name == "w_updraft":
        ax.set_yscale("log")
        y_label = r"vertical updraft [$\mathrm{m \, s^{-1}}$]"
        ens_label = "w ensemble"
    elif ens_var_name == "sigma_droplet_distribution":
        y_label = r"standard deviation DSD"
        ens_label = r"$\sigma$ ensemble"
    elif ens_var_name == "n_sd":
        ax.set_yscale("log")
        y_label = "number of super-particles"
        ens_label = r"$n_{sd}$ ensemble"

    ax.set_title(title_add + " " + title + " for " + ens_label, fontsize=ax_lab_fsize)
    ax.set_xlabel(x_label, fontsize=ax_lab_fsize)
    ax.set_ylabel(y_label, fontsize=ax_lab_fsize)
    ax.grid(visible=True)
    ax.legend(fontsize=ax_lab_fsize)

    return ax

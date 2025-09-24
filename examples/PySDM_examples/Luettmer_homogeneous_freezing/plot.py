import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from scipy.ndimage import histogram
import seaborn as sns

from PySDM import Formulae

formulae = Formulae(
    particle_shape_and_density="MixedPhaseSpheres",
)

# general plot settings
ax_lab_fsize = 15
tick_fsize = 15
# title_fsize = 15
# line_width = 2.5
T_frz_bins = np.linspace(-40, -34, num=60, endpoint=True)
T_frz_bins_kelvin = np.linspace(230, 240, num=100, endpoint=True)


def plot_thermodynamics_and_bulk(simulation, title_add="", show_conc=False):

    output = simulation["ensemble_member_outputs"][0]
    time = output["t"]
    T = np.asarray(output["T"])
    RH = np.asarray(output["RH"])
    RHi = np.asarray(output["RHi"])
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
        homogeneous_ice_nucleation_rate="Koop2000"
    ).homogeneous_ice_nucleation_rate
    koop_2000 = j_hom_rate.j_hom(T, d_a_w_ice)
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

    """ Temperture profile """
    ax = axs[0, 0]
    ax.plot(time, formulae.trivia.K2C(T), color="black", linestyle="-", label="T")
    ax.set_xlabel("time [s]", fontsize=ax_lab_fsize)
    ax.set_ylabel("temperature [°C]", fontsize=ax_lab_fsize)
    ax.legend(loc="upper left", fontsize=ax_lab_fsize)
    ax.tick_params(labelsize=tick_fsize)

    twin = ax.twinx()
    twin.plot(time, RH, color="red", linestyle="-", label="RH")
    twin.plot(time, RHi, color="blue", linestyle="-", label="RHi")
    twin.set_ylabel("relative humidity [%]", fontsize=ax_lab_fsize)
    twin.legend(loc="upper right", fontsize=ax_lab_fsize)
    twin.tick_params(labelsize=tick_fsize)
    twin.grid(visible=True)

    """ Water activity difference profile """
    lin_s_SP2023 = "--"
    lin_s_KM2016 = "-"
    if simulation["settings"]["hom_freezing"] == "Spichtinger2023":
        lin_s_SP2023 = "-"
        lin_s_KM2016 = "--"
    ax = axs[0, 1]
    # ax.plot(time, koop_2000, color="black", linestyle="-", label="Koop2000")
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

    """ Mass content and number concentration"""
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

    """ Mean radius """
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

        title = "Freezing method=" + simulation["settings"]["hom_freezing"]

        """ Freezing temperatures """
        hist = ax.hist(
            # formulae.trivia.K2C(T_frz),
            T_frz,
            # bins=T_frz_bins,
            bins=T_frz_bins_kelvin,
            density=True,
            cumulative=-1,
            alpha=1.0,
            histtype="step",
            linewidth=1.5,
        )

    ax.axvline(x=235, color="k", linestyle="--")
    ax.set_title(title, fontsize=ax_lab_fsize)
    ax.set_xlabel("freezing temperature [K]", fontsize=ax_lab_fsize)
    ax.set_ylabel("frequency", fontsize=ax_lab_fsize)
    ax.tick_params(labelsize=tick_fsize)

    return ax


def plot_freezing_temperatures_2d_histogram(histogram_data_dict):

    vertical_updrafts_bins = np.geomspace(0.05, 15, num=6, endpoint=True)

    hom_freezing_types = histogram_data_dict.keys()

    fig, axs = pyplot.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    axs = axs.ravel()
    i = 0
    for hom_freezing_type in hom_freezing_types:

        title = "Freezing method=" + hom_freezing_type

        ax = axs[i]
        T_frz = formulae.trivia.K2C(
            np.asarray(histogram_data_dict[hom_freezing_type]["T_frz_histogram_list"])
        )

        hist, x, y = np.histogram2d(
            T_frz,
            histogram_data_dict[hom_freezing_type]["w_updraft_histogram_list"],
            bins=(T_frz_bins, vertical_updrafts_bins),
        )
        y = np.log10(y)
        X, Y = np.meshgrid(x, y)

        hist = hist.T / sum(hist.flatten())

        c = ax.pcolor(X, Y, hist)
        fig.colorbar(c, ax=ax)

        ax.set_title(title, fontsize=ax_lab_fsize)
        ax.set_xlabel("freezing temperature [°C]", fontsize=ax_lab_fsize)
        ax.set_ylabel("vertical updraft [m/s]", fontsize=ax_lab_fsize)

        i += 1


def plot_freezing_temperatures_2d_histogram_seaborn(histogram_data_dict, title_add=""):

    # hom_freezing_types = ["KoopMurray2016", "Koop_Correction", "Koop2000"]
    sns.set_theme(style="ticks")

    hom_freezing_type = histogram_data_dict["hom_freezing_type"]
    # for i, hom_freezing_type in enumerate(hom_freezing_types):

    # T_frz = formulae.trivia.K2C(
    #     (np.asarray(histogram_data_dict["T_frz_histogram_list"]))
    # )
    T_frz = np.asarray(histogram_data_dict["T_frz_histogram_list"])

    if "w_updraft_histogram_list" in histogram_data_dict:
        w = histogram_data_dict["w_updraft_histogram_list"]
        y_label = r"vertical updraft [$\mathrm{m \, s^{-1}}$]"
    elif "n_ccn_histogram_list" in histogram_data_dict:
        w = histogram_data_dict["n_ccn_histogram_list"]
        y_label = r"ccn concentration [$\mathrm{m^{-3}}$]"
    elif "rc_max_histogram_list" in histogram_data_dict:
        w = np.asarray(histogram_data_dict["rc_max_histogram_list"]) * 1e6
        y_label = "(maximum) radius [µm]"

    # xlim = (-39.5, -34)
    xlim = (233, 241)
    h = sns.JointGrid(
        x=T_frz,
        y=w,
        xlim=xlim,
    )
    h.ax_joint.set(yscale="log")
    # if hom_freezing_type == "KoopMurray2016":
    #     x_pos_cbar = 0.75
    # else:
    x_pos_cbar = 0.75
    cax = h.figure.add_axes([x_pos_cbar, 0.55, 0.02, 0.2])
    h.plot_joint(
        sns.histplot,
        stat="density",
        binwidth=0.25,
        discrete=(False, False),
        pmax=0.8,
        cbar=True,
        cbar_ax=cax,
    )

    h.plot_marginals(
        sns.histplot,
        element="step",
    )
    h.set_axis_labels("freezing temperature [K]", y_label, fontsize=ax_lab_fsize)
    h.ax_joint.set_title(
        "Freezing: " + hom_freezing_type + title_add,
        pad=70,
        fontsize=ax_lab_fsize,
    )
    h.ax_marg_y.remove()

    return h


def plot_ensemble_bulk(
    simulations, var_name, ensemble_var, freezing_types, title_add=""
):

    ens_var, ens_var_name = ensemble_var
    len_ens_var = len(ens_var)

    for hom_freezing_type in freezing_types:
        var = np.zeros(len_ens_var)
        for i in range(len_ens_var):
            for simulation in simulations:
                if (
                    simulation["settings"]["hom_freezing"] == hom_freezing_type
                    and simulation["settings"][ens_var_name] == ens_var[i]
                ):
                    output = simulation["ensemble_member_outputs"][0]
                    if var_name == "freezing_fraction":
                        ni = np.asarray(output["ni"])[-1]
                        nc = np.asarray(output["ns"])[0]
                        var[i] = (1 - (nc - ni) / nc) * 100
                        # print("{:.2E}".format(nc),"{:.2E}".format(ni), var[i])
                        # quit()
                    else:
                        var[i] = np.asarray(output[var_name])[-1]

        pyplot.scatter(var, ens_var, label=hom_freezing_type)

    if var_name == "ni":
        pyplot.xscale("log")
        x_label = r"$n_{i} \, [\mathrm{kg^{-1}}$]"
        title = "Ice number concentrations"
        pyplot.xlim(1e6, 1e10)

    if var_name == "IWC":
        pyplot.xscale("log")
        x_label = r"mass content [$\mathrm{kg \, kg^{-1}}$]"
        title = "Ice mass content"

    if var_name == "freezing_fraction":
        title = "frozen fraction of real droplets "
        x_label = r"$n_{frz} \, [\mathrm{\%}$]"

    if ens_var_name == "N_dv_droplet_distribution":
        pyplot.yscale("log")
        y_label = r"$n_{ccn} \, [\mathrm{m^{-3}}$]"

    if ens_var_name == "w_updraft":
        pyplot.yscale("log")
        y_label = r"w [$\mathrm{m \, s^{-1}}$]"

    pyplot.title(title + title_add, fontsize=ax_lab_fsize)
    pyplot.xlabel(x_label, fontsize=ax_lab_fsize)
    pyplot.ylabel(y_label, fontsize=ax_lab_fsize)
    pyplot.legend(fontsize=ax_lab_fsize)

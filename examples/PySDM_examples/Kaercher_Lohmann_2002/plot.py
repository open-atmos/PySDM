import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import boxplot

from reference import bulk_model_reference, critical_supersaturation_spich2023

import json
import numpy as np

# general plot settings
ax_lab_fsize = 15
tick_fsize = 15
title_fsize = 15

kg_to_µg = 1.e9
m_to_µm = 1.e6

def plot_size_distribution(r_wet, r_dry, N, setting, pp):
    r_wet, r_dry = r_wet * m_to_µm, r_dry * m_to_µm

    title = f"N0: {setting.N_dv_solution_droplet * 1e-6:.2E} cm-3 \
    R0: {setting.r_mean_solution_droplet * m_to_µm:.2E} µm \
    $\sigma$: {setting.sigma_solution_droplet:.2f} \
    Nsd: {setting.n_sd:d}  $\kappa$: {setting.kappa:.2f}"

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(r_dry, N, color="red", label="dry")
    ax.scatter(r_wet, N, color="blue", label="wet")
    ax.set_title(title)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(5.e-3, 5.e0)
    ax.set_yscale('log')
    ax.set_xlabel(r"radius [µm]")
    ax.set_ylabel("multiplicty")

    pp.savefig()


def plot_evolution( pp, output ):
    ln = 2.5

    extra_panel = True

    time = output["t"]
    temperature = np.asarray(output["T"])
    z = output["z"]

    # rh = output["RH"]
    rhi = np.asarray(output["RHi"]) / 100.

    # rhi_crit_old = critical_supersaturation(temperature)
    rhi_crit = critical_supersaturation_spich2023(temperature)

    lwc = np.asarray(output["LWC"]) * kg_to_µg
    iwc = abs(np.asarray(output["IWC"])) * kg_to_µg

    ns = output["ns"]
    ni = output["ni"]
    rs = output["rs"]
    ri = output["ri"]

    frozen = output["frozen"]
    r_wet_init = np.asarray(output["r_wet"]) * m_to_µm
    multiplicities = output["multiplicity"]
    multiplicities_unfrozen = np.ma.masked_array(multiplicities, mask=frozen)
    multiplicities_frozen = np.ma.masked_array(multiplicities, mask=np.logical_not(frozen))

    if extra_panel:
        fig, axs = pyplot.subplots(3, 2, figsize=(10, 15), sharex=False)
    else:
        fig, axs = pyplot.subplots(2, 2, figsize=(10, 10), sharex=False)


    x_limit = [0, np.amax(time)]

    # title = (f"w: {setting.w_updraft:.2f} m s-1 T0: {setting.initial_temperature:.2f} K "
    #          f"Nsd: {setting.n_sd:d} rate: ") + setting.rate
    # fig.suptitle(title)

    axTz = axs[0, 0]
    axTz.plot(
        time, temperature, color="red", linestyle="-", label="T", linewidth=ln
    )

    twin = axTz.twinx()
    twin.plot(
        time, z, color="black", linestyle="-", label="dz", linewidth=ln
    )
    twin.set_ylabel("vertical displacement [m]", fontsize=ax_lab_fsize)
    twin.set_ylim(-5, np.amax(z) + 100)
    axTz.grid(True)
    axTz.set_ylim(190, 225)
    axTz.legend(loc='upper right',fontsize=ax_lab_fsize)
    axTz.set_ylabel("temperature [K]", fontsize=ax_lab_fsize)
    twin.legend(loc='upper left',fontsize=ax_lab_fsize)
    axTz.set_xlabel("time [s]", fontsize=ax_lab_fsize)
    axTz.tick_params(labelsize=tick_fsize)
    twin.tick_params(labelsize=tick_fsize)
    axTz.set_title("(a) air parcel ascent", fontsize=title_fsize)
    axTz.set_xlim(x_limit)

    axRH = axs[0, 1]
    # axRH.plot(
    #     time, rh, color="blue", linestyle="-", label="water", linewidth=ln
    # )
    axRH.plot(
        time, rhi, color="red", linestyle="-", label="ice", linewidth=ln
    )
    axRH.plot(
        time, rhi_crit, color="black", linestyle="-", label="crit", linewidth=ln
    )
    # axRH.plot(
    #     time, rhi_crit_old, color="black", linestyle="--", label="crit_old", linewidth=ln
    # )
    axRH.grid(True)
    axRH.legend(loc='lower right',fontsize=ax_lab_fsize)
    axRH.set_xlabel("time [s]", fontsize=ax_lab_fsize)
    axRH.set_xlim(x_limit)
    axRH.set_ylabel("supersaturation ratio", fontsize=ax_lab_fsize)
    axRH.set_ylim(1, 1.6)
    axRH.tick_params(labelsize=tick_fsize)
    axRH.set_title("(b) supersaturation", fontsize=title_fsize)

    axN = axs[1, 0]

    axN.plot(
        time, ns, color="blue", linestyle="-", label="solution droplet", linewidth=ln)
    axN.plot(
        time, ni, color="red", linestyle="-", label="ice", linewidth=ln)

    axN.set_yscale('log')
    axN.grid(True)
    axN.legend(fontsize=ax_lab_fsize)
    axN.set_ylim(1.e-3, 5.e4)
    axN.set_xlabel("time [s]", fontsize=ax_lab_fsize)
    axN.set_xlim(x_limit)
    axN.set_ylabel(r"number concentration [$\mathrm{cm^{-3}}$]"
                   , fontsize=ax_lab_fsize)
    axN.tick_params(labelsize=tick_fsize)
    axN.set_title("(c) bulk number concentration", fontsize=title_fsize)

    axR = axs[1, 1]

    axR.plot(
        time, rs, color="blue", linestyle="-", label="solution droplet", linewidth=ln)
    axR.plot(
        time, ri, color="red", linestyle="-", label="ice", linewidth=ln)

    axR.grid(True)
    axR.legend(fontsize=ax_lab_fsize)
    axR.set_yscale('log')
    axR.set_ylim(1.e-2, 1.e2)
    axR.set_xlabel("time [s]", fontsize=ax_lab_fsize)
    axR.set_xlim(x_limit)
    axR.set_ylabel(r"mean radius [µm]", fontsize=ax_lab_fsize)
    axR.tick_params(labelsize=tick_fsize)
    axR.set_title("(d) mean radius", fontsize=title_fsize)

    if extra_panel:

        axWC = axs[2, 0]

        axWC.plot(
            time, lwc, color="blue", linestyle="-", label="water", linewidth=ln
        )
        axWC.plot(
            time, iwc, color="red", linestyle="-", label="ice", linewidth=ln
        )
        axWC.set_title("(e) mass content", fontsize=title_fsize)
        axWC.set_yscale('log')
        axWC.legend(fontsize=ax_lab_fsize)
        axWC.tick_params(labelsize=tick_fsize)
        axWC.set_xlim(x_limit)
        axWC.set_ylim(1.e0, 5.e4)
        axWC.set_xlabel("time [s]", fontsize=ax_lab_fsize)
        axWC.set_ylabel(r"mass content [$\mathrm{\mu g \, kg^{-1}}$]"
                        , fontsize=ax_lab_fsize)
        axWC.grid(True)

        axFrz = axs[2, 1]

        axFrz.scatter(r_wet_init, multiplicities_unfrozen, color="black", alpha=0.5, label="unfrozen")
        axFrz.scatter(r_wet_init, multiplicities_frozen, color="blue", label="frozen")
        axFrz.set_title("(f) frozen super particles", fontsize=title_fsize)
        axFrz.legend(fontsize=ax_lab_fsize)
        axFrz.tick_params(labelsize=tick_fsize)
        axFrz.set_xscale('log')
        axFrz.set_xlim(5.e-3, 5.e-1)
        axFrz.set_yscale('log')
        axFrz.set_xlabel(r"initial radius [µm]", fontsize=ax_lab_fsize)
        axFrz.set_ylabel("multiplicty", fontsize=ax_lab_fsize)
        # axFrz.grid(True)

    fig.tight_layout()
    pp.savefig()


def plot_ensemble(pp, outputs, T0, ens_member, only_ni = False, title = None):



    x_label = "number of super particles"

    ni_bulk_ref = bulk_model_reference(T0) * 1e-6

    nx = len(outputs)

    x_array =  np.array([50, 100, 500, 1000, 5000, 10000, 50000, 100000])
    x_string_array = []
    for x in x_array:
        x_string_array.append(f"{x:.0e}")

    ni_arr = np.empty( ( ens_member,np.shape(x_array)[0])  )
    ri_arr = np.empty((ens_member, np.shape(x_array)[0]))
    frozen_fraction_arr = np.empty((ens_member, np.shape(x_array)[0]))
    min_frozen_r_arr = np.empty((ens_member, np.shape(x_array)[0]))

    jdx = 0
    idx_ref = 0
    for i in range(nx):
        output = outputs[i]
        n_sd = output["n_sd"]
        idx = np.nonzero(x_array==n_sd)[0][0]

        if idx > idx_ref:
            idx_ref = idx
            jdx = 0

        ni_arr[jdx,idx] = output["ni"][-1]
        ri_arr[jdx,idx] = output["ri"][-1]
        frozen = output["frozen"]
        frozen_fraction_arr[jdx,idx]  = np.sum(frozen) / np.size(frozen)
        min_frozen_r_arr[jdx,idx]  = m_to_µm * np.amin(
            np.ma.masked_array(output["r_dry"],
                               mask=np.logical_not(frozen))
        )
        jdx += 1

    if only_ni:
        fig, ax = pyplot.subplots(1, 1, figsize=(5, 5))
    else:
        fig, axs = pyplot.subplots(2, 2, figsize=(10, 10), sharex=False)
    # fig.suptitle(title)

    if only_ni:
        if title is not None:
            ax.set_title(title, fontsize=title_fsize)
        ax.boxplot(ni_arr, tick_labels=x_string_array)
        ax.set_yscale('log')
        ax.set_ylim(1.e-2, 1.e2)
        ax.axhline(ni_bulk_ref, c="r")
        ax.set_ylabel(r"ice number concentration [$\mathrm{cm^{-3}}$]"
                       , fontsize=ax_lab_fsize)
        ax.set_xlabel(x_label, fontsize=ax_lab_fsize)
        ax.tick_params(axis='y', labelsize=tick_fsize)
    else:
        axN = axs[0, 0]
        axN.set_title("(a) nucleated number concentration", fontsize=title_fsize)
        axN.boxplot(ni_arr,tick_labels=x_string_array)
        axN.set_yscale('log')
        axN.set_ylim(1.e-2, 1.e2)
        axN.axhline(ni_bulk_ref, c="r")
        axN.set_ylabel(r"ice number concentration [$\mathrm{cm^{-3}}$]"
                       , fontsize=ax_lab_fsize)
        axN.set_xlabel(x_label, fontsize=ax_lab_fsize)
        axN.tick_params(axis='y', labelsize=tick_fsize)

        axR = axs[0, 1]
        axR.set_title("(b) mean radius", fontsize=title_fsize)
        axR.boxplot(ri_arr,tick_labels=x_string_array)
        axR.set_xlabel(x_label)
        axR.set_yscale('log')
        axR.set_ylim(5.e-2, 1.e2)
        axR.set_ylabel(r"ice mean radius[µm]"
                       , fontsize=ax_lab_fsize)
        axR.tick_params(axis='y', labelsize=tick_fsize)

        axF = axs[1, 0]
        axF.set_title("(c) frozen fraction", fontsize=title_fsize)
        axF.boxplot(frozen_fraction_arr,tick_labels=x_string_array)
        axF.set_xlabel(x_label, fontsize=ax_lab_fsize)
        axF.set_yscale('log')
        axF.set_ylim(1.e-5, 1)
        axF.set_ylabel(r"fraction of frozen super particles"
                       , fontsize=ax_lab_fsize)
        axF.tick_params(axis='y', labelsize=tick_fsize)

        axM = axs[1, 1]
        axM.set_title("(d) radius of smallest frozen droplet", fontsize=title_fsize)
        axM.boxplot(min_frozen_r_arr,tick_labels=x_string_array)
        axM.set_xlabel(x_label, fontsize=ax_lab_fsize)
        axM.set_yscale('log')
        axM.set_ylim(5.e-3, 5.e0)
        axM.set_ylabel(r"minimum radius of frozen droplets [µm]"
                       , fontsize=ax_lab_fsize)
        axM.tick_params(axis='y', labelsize=tick_fsize)

    fig.tight_layout()
    pp.savefig()



# plot super particle ensemble
def plot_ensemble_simulation(file_name, only_ni=False, title=None):

    plot_name = file_name.replace("json","pdf")

    with open(file_name, 'r') as f:
        data = json.load(f)
    outputs = data["outputs"]
    number_of_ensemble_runs = data["number_of_ensemble_runs"]
    T0 = data["initial_temperature"]

    pp = PdfPages(plot_name)
    plot_ensemble(pp, outputs, T0, number_of_ensemble_runs, only_ni=only_ni, title = title)
    pp.close()

# filename= "ensemble_nsd_25_dsd_0_T0_220.json"
# plot_ensemble_simulation(filename)
filename= "ensemble_nsd_25_dsd_0_T0_220_lin.json"
plot_ensemble_simulation(filename, only_ni=True, title="linear sampling")
filename= "ensemble_nsd_25_dsd_0_T0_220_limit.json"
plot_ensemble_simulation(filename, only_ni=True, title="limited sampling")




# plot super particle ensemble
def plot_simulation_evolution(file_name):

    plot_name = file_name.replace("json","pdf")

    with open(file_name, 'r') as f:
        data = json.load(f)
    outputs = data["outputs"]

    pp = PdfPages(plot_name)
    for output in outputs:
        plot_evolution(pp, output)
    pp.close()

# filename="ensemble_1_dsd_1_T0_220_nsd_50000highoutput.json"
# plot_simulation_evolution(filename)

def plot_size_distribution_discretisation():

    target_n_sd = 50
    def get_r_and_m(file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
        for output in data["outputs"]:
            if output["n_sd"][0] == target_n_sd:
                return( np.array(output["r_wet"][0]),
                        np.array(output["multiplicity"][0]) )

    file_name = "ensemble_nsd_1_dsd_0_T0_220_nsd_50.json"
    dsd1 = get_r_and_m(file_name)

    file_name = "ensemble_nsd_1_dsd_1_T0_220_nsd_50.json"
    dsd2 = get_r_and_m(file_name)

    file_name = "ensemble_nsd_1_dsd_2_T0_220_nsd_50.json"
    dsd3 = get_r_and_m(file_name)

    file_name = "ensemble_nsd_1_dsd_0_T0_220_nsd_50_lin.json"
    dsd1_lin = get_r_and_m(file_name)

    # file_name    = "ensemble_1_dsd_0_T0_220_nsd_50_RH16.json"
    # dsd1_16 = get_r_and_m(file_name)
    # file_name = "ensemble_1_dsd_0_T0_220_nsd_50_RH10.json"
    # dsd1_10 = get_r_and_m(file_name)

    file_name = "ensemble_1_dsd_0_T0_190_nsd_50.json"
    dsd_1_190 = get_r_and_m(file_name)

    file_name = "ensemble_1_dsd_0_T0_200_nsd_50.json"
    dsd_1_200 = get_r_and_m(file_name)

    file_name = "ensemble_1_dsd_0_T0_220_nsd_50.json"
    dsd_1_220 = get_r_and_m(file_name)

    file_name = "ensemble_nsd_1_dsd_0_T0_220_nsd_50_limit.json"
    dsd1_lim  =  get_r_and_m(file_name)




    fig, axs = pyplot.subplots(2, 2, figsize=(10, 10), sharex=False)
    # fig.suptitle("Solution droplet size distributions")

    ax = axs[0, 0]

    xaxis_label = r"droplet radius r [µm]"

    ax.scatter(dsd1[0] * m_to_µm, dsd1[1], color="black", label="DSD1")
    ax.scatter(dsd2[0] * m_to_µm, dsd2[1], color="red", label="DSD2")
    ax.scatter(dsd3[0] * m_to_µm, dsd3[1], color="blue", label="DSD3")
    ax.tick_params(labelsize=tick_fsize)
    ax.set_xscale('log')
    ax.set_xlim(5.e-3, 5.e0)
    ax.set_ylim(5e7, 2e12)
    ax.set_yscale('log')
    ax.set_xlabel(xaxis_label,fontsize=ax_lab_fsize)
    ax.set_ylabel("multiplicty",fontsize=ax_lab_fsize)
    ax.set_title("(a) logarithmic sampling",fontsize=title_fsize)
    ax.legend(fontsize=ax_lab_fsize)

    ax = axs[0, 1]
    ax.scatter(dsd1_lim[0] * m_to_µm, dsd1_lim[1], color="black", label="DSD1")
    ax.tick_params(labelsize=tick_fsize)
    ax.set_xscale('log')
    ax.set_xlim(5.e-3, 5.e0)
    ax.set_ylim(1e7, 2e12)
    ax.set_yscale('log')
    ax.set_xlabel(xaxis_label, fontsize=ax_lab_fsize)
    ax.set_ylabel("multiplicty", fontsize=ax_lab_fsize)
    ax.set_title("(b) limited logarithmic sampling", fontsize=title_fsize)
    ax.legend(fontsize=ax_lab_fsize)


    ax = axs[1, 0]
    ax.scatter(dsd1_lin[0] * m_to_µm, dsd1_lin[1], color="black", label="DSD1")
    ax.tick_params(labelsize=tick_fsize)
    ax.set_xscale('log')
    ax.set_xlim(5.e-3, 5.e0)
    ax.set_ylim(1e7, 2e12)
    ax.set_yscale('log')
    ax.set_xlabel(xaxis_label,fontsize=ax_lab_fsize)
    ax.set_ylabel("multiplicty",fontsize=ax_lab_fsize)
    ax.set_title("(c) linear sampling",fontsize=title_fsize)
    ax.legend(fontsize=ax_lab_fsize)

    ax = axs[1, 1]

    # ax.scatter(dsd1_10[0] * m_to_µm, dsd1_10[1], color="black", label="T=220K")
    # ax.scatter(dsd1_16[0] * m_to_µm, dsd1_16[1], color="red", label="T=200K")
    ax.scatter(dsd_1_220[0] * m_to_µm, dsd_1_220[1], color="black", label="T=220K")
    ax.scatter(dsd_1_200[0] * m_to_µm, dsd_1_200[1], color="red", label="T=200K")
    ax.scatter(dsd_1_190[0] * m_to_µm, dsd_1_190[1], color="blue", label="T=190K")
    ax.tick_params(labelsize=tick_fsize)
    ax.set_xscale('log')
    ax.set_xlim(5.e-3, 5.e0)
    ax.set_ylim(5e7, 2e12)
    ax.set_yscale('log')
    ax.set_xlabel(xaxis_label, fontsize=ax_lab_fsize)
    ax.set_ylabel("multiplicty", fontsize=ax_lab_fsize)
    ax.set_title("(d) temperature dependence", fontsize=title_fsize)
    ax.legend(fontsize=ax_lab_fsize)

    fig.tight_layout()
    plt.savefig("size_distributions.pdf")



def plot_ni_as_function_of_w():

    dsd_list = np.array([0])
    initial_temperatures = np.array([196., 216., 236.])
    updrafts = np.array([0.05, 0.1, 0.5, 1., 5., 10.])

    dim_size = ( np.shape(dsd_list)[0], np.shape(initial_temperatures)[0], np.shape(updrafts)[0] )
    ni_sdm = np.zeros(dim_size)



    for i in range(dim_size[0]):
        for j in range(dim_size[1]):
            T0 = initial_temperatures[j]
            for k in range(dim_size[2]):
                w = updrafts[k]
                filename = "ensemble_1_dsd_"+str(i)+f"_T0_{T0:.0f}"+f"_W_{w:.2f}"+"_nsd_10.json"
                # print(filename)

                with open(filename, 'r') as f:
                    data = json.load(f)

                T0_sim = data['initial_temperature']
                w_sim   = data['w_updraft']
                if T0 == T0_sim and w == w_sim:
                    output = data['outputs'][0]
                    ni_last = output["ni"][-1]
                    print( w, T0, ni_last )
                    ni_sdm[i,j,k] = ni_last

    print(ni_sdm)

    fig, axs = pyplot.subplots(1, 3, figsize=(15, 5), sharex=False)

    for i in range(1):
        ax = axs[i]

        if i == 0:
            ax.set_title("(a) hom. nucleation for DSD 1",fontsize=title_fsize)
        if i == 1:
            ax.set_title("(b) hom. nucleation for DSD 2", fontsize=title_fsize)
        if i == 2:
            ax.set_title("(c) hom. nucleation for DSD 3", fontsize=title_fsize)

        for j in range(dim_size[1]):
         ax.scatter(updrafts, ni_sdm[i,j,:],label=f"T0={initial_temperatures[j]:.0f}K")

        ax.tick_params(labelsize=tick_fsize)
        ax.set_xscale('log')
        ax.set_xlabel(r"vertical updraft [$\mathrm{m \, s^{-1}}$]"
                       , fontsize=ax_lab_fsize)
        ax.set_yscale('log')
        ax.set_ylim(1.e-1, 8.e3)
        ax.set_ylabel(r"ice number concentration [$\mathrm{cm^{-3}}$]"
                       , fontsize=ax_lab_fsize)
        ax.legend(fontsize=ax_lab_fsize,loc="lower right")

    plt.tight_layout
    plt.savefig("w_ni_plot.pdf")
# plot_size_distribution_discretisation()

# print( critical_supersaturation(220.) )
# plot_ni_as_function_of_w()
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


from settings import settings
from simulation import Simulation
from reference import critical_supersaturation
from PySDM.physics.constants import si


kg_to_µg = 1.e9
m_to_µm  = 1.e6

def plot_size_distribution(r_wet, r_dry, N, setting, pp ):

    r_wet, r_dry = r_wet * m_to_µm, r_dry * m_to_µm

    title = f"N0: {setting.N_dv_solution_droplet*1e-6:.2E} cm-3 \
    R0: {setting.r_mean_solution_droplet*m_to_µm:.2E} µm \
    $\sigma$: {setting.sigma_solution_droplet:.2f} \
    Nsd: {setting.n_sd:d}  $\kappa$: {setting.kappa:.2f}"

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.scatter( r_dry, N, color="red", label = "dry" )
    ax.scatter( r_wet, N, color="blue", label = "wet" )
    ax.set_title(title)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlim(5.e-3,5.e0)
    ax.set_yscale('log')
    ax.set_xlabel(r"radius [µm]")
    ax.set_ylabel("multiplicty")

    pp.savefig()


def plot( output, setting, pp ):


    time = output["t"]
    temperature = np.asarray(output["T"])
    z = output["z"]
    
    rh = output["RH"]
    rhi = output["RHi"] 
    rhi_crit = critical_supersaturation(temperature) * 100.


    print(f"{rh=},{rhi=},{rhi_crit=}")

    lwc = np.asarray(output["LWC"]) * kg_to_µg
    iwc = abs(np.asarray(output["IWC"])) * kg_to_µg
    twc = lwc + iwc
    qv = np.asarray(output["qv"]) * kg_to_µg

    print(f"{lwc=},{iwc=},{twc=},{qv=}")

    ns = output["ns"]
    ni = output["ni"]
    rs = output["rs"]
    ri = output["ri"]
    print(f"{ns=},{ni=},{rs=},{ri=},")


    fig, axs = pyplot.subplots(3, 2, figsize=(10, 10), sharex=True)


    title = f"w: {setting.w_updraft:.2f} m s-1 T0: {setting.initial_temperature:.2f} K Nsd: {setting.n_sd:d} \
    rate: " + setting.rate

    fig.suptitle(title)

    axTz = axs[0,0]

    axTz.plot(
        time, z, color="black", linestyle="-", label="dz", linewidth=5
    )
    axTz.set_ylabel("vertical displacemet [m]")
    axTz.set_ylim(-5, 1000)
    twin = axTz.twinx()
    twin.plot(
        time, temperature, color="red", linestyle="-", label="T", linewidth=5
    )
    twin.set_ylim(190, 250)
    twin.legend(loc='upper right')
    twin.set_ylabel("temperature [K]")
    axTz.legend(loc='upper left')
    axTz.set_xlabel("time [s]")
  

    axRH = axs[0,1]
    
    axRH.plot(
        time, rh, color="blue", linestyle="-", label="water", linewidth=5
    )
    axRH.plot(
        time, rhi, color="red", linestyle="-", label="ice", linewidth=5
    )
    axRH.plot(
        time, rhi_crit, color="black", linestyle="-", label="crit", linewidth=5
      )
    axRH.legend()
    axRH.set_xlabel("time [s]")
    axRH.set_ylabel("relative humidity [%]")
    axRH.set_ylim(50, 200)



    axWC = axs[1,0]

    
    axWC.plot(
        time, twc, color="black", linestyle="--", label="total", linewidth=5
    )
    axWC.plot(
        time, lwc, color="blue", linestyle="-", label="water", linewidth=5
    )
    axWC.plot(
        time, iwc, color="red", linestyle="-", label="ice", linewidth=5
    )
    #axWC.set_yscale('log')
    axWC.legend()
    axWC.set_ylim(-0.5, 100)
    axWC.set_xlabel("time [s]")
    axWC.set_ylabel(r"mass content [$\mathrm{\mu g \, kg^{-1}}$]")



    axN = axs[1,1]


    axN.plot(
        time, ns, color="blue", linestyle="-", label="droplet", linewidth=5 )
    axN.plot(
        time, ni, color="red", linestyle="-", label="ice", linewidth=5 )

    axN.legend()
    axN.set_ylim(-0.5, 3000)
    axN.set_xlabel("time [s]")
    axN.set_ylabel(r"number concentration [$\mathrm{cm^{-3}}$]")

   
    axR = axs[2,0]


    axR.plot(
        time, rs, color="blue", linestyle="-", label="droplet", linewidth=5 )
    axR.plot(
        time, ri, color="red", linestyle="-", label="ice", linewidth=5 )


    axR.legend()
    #axR.set_ylim(-0.5, 3000)
    axR.set_xlabel("time [s]")
    axR.set_ylabel(r"mean radius [µm]")


    fig.tight_layout() 
    pp.savefig()

general_settings = {"n_sd": 1000, "T0": 220 * si.kelvin, "w_updraft": 10 * si.centimetre / si.second}
distributions = ({"N_dv_solution_droplet": 2500 / si.centimetre**3, \
                  "r_mean_solution_droplet": 0.055 * si.micrometre, \
                  "sigma_solution_droplet": 1.6},
                 {"N_dv_solution_droplet": 8600 / si.centimetre**3, \
                  "r_mean_solution_droplet": 0.0275 * si.micrometre, \
                  "sigma_solution_droplet": 1.3},
                 {"N_dv_solution_droplet": 2000 / si.centimetre**3, \
                  "r_mean_solution_droplet": 0.11 * si.micrometre, \
                  "sigma_solution_droplet": 2.},
                 )

pp = PdfPages( "hom_freezing_for_size_distributions.pdf" )

for distribution in distributions:

    setting = settings( **{**general_settings, **distribution} )
    model = Simulation(setting)
    output = model.run()

    plot_size_distribution( model.r_wet, model.r_dry, model.multiplicities, setting, pp)
    plot( output, setting, pp )

pp.close()


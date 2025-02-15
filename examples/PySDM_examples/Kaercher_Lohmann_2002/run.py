from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from settings import setups
from simulation import Simulation
from reference import critical_supersaturation


kgtoug = 1.e9


def plot( output, setting, pp ):


    time = output["t"]
    temperature = np.asarray(output["T"])
    z = output["z"]
    
    rh = output["RH"]
    rhi = output["RHi"] 
    rhi_crit = critical_supersaturation(temperature) * 100.


    print(f"{rh=},{rhi=},{rhi_crit=}")

    lwc = np.asarray(output["LWC"]) * kgtoug
    iwc = abs(np.asarray(output["IWC"])) * kgtoug
    twc = lwc + iwc
    qv = np.asarray(output["qv"]) * kgtoug

    print(f"{lwc=},{iwc=},{twc=},{qv=}")

    ns = output["ns"]
    ni = output["ni"]
    rs = output["rs"]
    ri = output["ri"]
    print(f"{ns=},{ni=},{rs=},{ri=},")


    fig, axs = pyplot.subplots(3, 2, figsize=(10, 10), sharex=True)


    title = f"w: {setting.w_updraft:.2f} m s-1    T0: {setting.initial_temperature:.2f} K   Nsd: {setting.n_sd:d}   $\kappa$: {setting.kappa:.2f} rate: " + setting.rate
    # $\mathrm{m \, s^{-1}}$
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

pp = PdfPages( "hom_freezing.pdf" )
    

for setting in setups:


    model = Simulation(setting)

    output = model.run()

    plot( output, setting, pp )


    
pp.close()

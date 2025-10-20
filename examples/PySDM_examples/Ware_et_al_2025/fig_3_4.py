import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.colors import ListedColormap
import cmcrameri as cm
import seaborn as sns
import colormaps as cmaps
from matplotlib.ticker import ScalarFormatter
from cmap import Colormap
import matplotlib.colors as colors
from PySDM_examples.Ware_et_al_2025.box_simulation import main


regular_data,adaptive_data,type_matrix = main(plot=False, save=".")


global_droplets_norm_error = max(
    max(max(regular_data["ConstantMultiplicity"]["Error"])),
    max(max(regular_data["Logarithmic"]["Error"])),
    max(max(regular_data["Linear"]["Error"])),
)
global_droplets_norm_time = max(
    max(max(adaptive_data["ConstantMultiplicity"]["MeanTime"])),
    max(max(adaptive_data["Logarithmic"]["MeanTime"])),
    max(max(adaptive_data["Linear"]["MeanTime"])),
)

# Example parameters
log2_Ns = [13, 14, 15, 16, 17, 18,19]
dts = [20.0, 10.0, 5.0, 2.0, 1.0]
init_names = ["ConstantMultiplicity", "Logarithmic", "Linear"]
init_py = init_names  # Assuming `init_py` is equivalent to `init_names` in this context


#%%
#create mosaic layout with init_names+rows
fig, axes = plt.subplot_mosaic([
        [init+"_Error" for init in init_names],
        [init+"_MeanTime" for init in init_names],
        [init+"_Deficit" for init in init_names],
    ],sharex=True, sharey=True, figsize=(8, 4), constrained_layout=True)
# Create heatmaps for regular data

#replace all 0s with NaN
# np.where(np.array(regular_data).any(0) == 0, np.nan, regular_data)

vminmx = {
    "Error": [np.min([regular_data[i]["Error"] for i in init_names]),
                np.max([regular_data[i]["Error"] for i in init_names])],
    "MeanTime": [np.min([regular_data[i]["MeanTime"] for i in init_names]),
                np.max([regular_data[i]["MeanTime"] for i in init_names])],
    "Deficit": [1,
                 np.max([regular_data[i]["Deficit"] for i in init_names])],
}
mult = {
    "Error": 1e3,
    "MeanTime": 1/vminmx["MeanTime"][1],
    "Deficit": 1,
}
labels ={
    "Error": f"RMSE at t=3600s\n[g/m$^3$/unit ln(r)]",
    "MeanTime": "WallTime \n normalized",
    "Deficit": "Collision Deficit\n$[s^{{-1}} m^{{-3}}$]",
}
init_naming = {
    "ConstantMultiplicity": "constant-multiplicity",
    "Logarithmic": "uniform-in-log($r$)",
    "Linear": "uniform-in-$r$",
}

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0, 0))  # Force scientific notation for all values


for row in ["Error", "MeanTime","Deficit"]:        
    vmin = vminmx[row][0] * mult[row]
    vmax = vminmx[row][1] * mult[row]

    for init in init_names:
        ax = axes[f"{init}_{row}"]
        data = np.array(regular_data[init][row])#,axis=0)
        data = np.where(data == 0, np.nan, data)*mult[row]
        # Use Matplotlib contour plot
        bounds = np.linspace(vmin, vmax, 20) if row != "Deficit" else \
            np.logspace(np.log10(vmin), np.log10(vmax), 20)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

        im = ax.imshow(
                data,     
                # ax=ax, 
                norm=norm,               
                # norm=LogNorm(vmin=vmin,vmax=vmax) if row =="Deficit" else \
                #     Normalize(vmin=vmin, vmax=vmax),
                cmap=plt.get_cmap(Colormap('crameri:oslo_r').to_mpl(),10),
                aspect="auto",
                # cbar=False,# if init == init_names[-1] else False,
            )
        if init == init_names[0]:
            cbar = fig.colorbar(
                im, 
                ax=axes[f"{init_names[-1]}_{row}"], 
                orientation="vertical", 
                aspect=5,
                label=labels[row],
                # ticks=bounds[::4],  # Add ticks at the boundaries
                format='%.2f' if row != "Deficit" else formatter, 
            )
            cbar.ax.yaxis.set_offset_position('left')

        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color("black")

        if row == "Error":
            ax.set_title(f"{init_naming[init]}")
        if row =="Deficit":
            ax.set_xlabel("$N_s$")
        if init == init_names[0]:
            ax.set_ylabel(f"$\Delta t$ [s]")
        ax.set_xticks(np.arange(len(log2_Ns)))
        ax.set_xticklabels([f"$2^{{{n}}}$" for n in log2_Ns])
        ax.set_yticks(np.arange(len(dts)))
        ax.set_yticklabels(dts, rotation=0)

    cbar.set_label(labels[row])
plt.savefig("Heatmaps.pdf")
plt.show()
# %%

\
fig, axes = plt.subplot_mosaic([
        [init+"_Error" for init in init_names],
        [init+"_MeanTime" for init in init_names],
    ],sharex=True, sharey=True, figsize=(8, 3), constrained_layout=True)
for row in ["Error", "MeanTime"]:
    vmin = vminmx[row][0] * mult[row]
    vmax = vminmx[row][1] * mult[row]

    for init in init_names:
        ax = axes[f"{init}_{row}"]
        data = np.array(adaptive_data[init][row])#,axis=0)
        data = np.where(data == 0, np.nan, data)*mult[row]

        bounds = np.linspace(vmin, vmax, 20)
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

        contour = ax.imshow(
                data,    
                # levels = 15,   
                norm = norm,             
                # norm=LogNorm(vmin=vmin,vmax=vmax,ncolors=4) if row =="Deficit" else \
                #     Normalize(vmin=vmin, vmax=vmax, ncolors=2),
                cmap=plt.get_cmap(Colormap('crameri:oslo_r').to_mpl(),20),
                aspect="auto",
                # ('oslo', Nlevels=15),
            )
        ax.invert_yaxis()

        if init == init_names[0]:
            cbar = fig.colorbar(
                contour,
                ax=axes[f"{init_names[-1]}_{row}"], 
                orientation="vertical", 
                aspect=5, 
                # pad=0.1, 
                label=labels[row],
                ticks=bounds[::4], 
                format='%.2f' if row != "Deficit" else formatter,  #
            )
            cbar.ax.yaxis.set_offset_position('left')

        if row == "Error":
            ax.set_title(f"{init_naming[init]}")
        if row =="MeanTime":
            ax.set_xlabel("$N_s$")
        if init == init_names[0]:
            ax.set_ylabel(f"$\Delta t$ [s]")
        ax.set_xticks(np.arange(len(log2_Ns)))
        ax.set_xticklabels([f"$2^{{{n}}}$" for n in log2_Ns])
        ax.set_yticks(np.arange(len(dts)))
        ax.set_yticklabels(dts)
# plt.tight_layout()
plt.savefig("AdaptiveHeatmaps.pdf")
plt.show()


################################################################
# make LoWo21 Figure 2
# r_min, fraction raindrop mass evaporated as functions of RH
################################################################
import numpy as np
import matplotlib.pyplot as plt
import os

# load results
root_dir = os.path.dirname(os.path.abspath(__file__))
RHs = np.load(os.path.join(root_dir, "RHs.npy"))
r0grid = np.load(os.path.join(root_dir, "r0grid.npy"))
RHgrid = np.load(os.path.join(root_dir, "RHgrid.npy"))
m_frac_evap = np.load(os.path.join(root_dir, "m_frac_evap.npy"))
r_mins = np.load(os.path.join(root_dir, "r_mins.npy"))

i_RH75 = 29  # index for RH=0.75

# make figure 2
# only put colorbar on lower panel, still line up x-axes
f, axs = plt.subplots(
    2,
    2,
    sharex="col",
    figsize=(6, 7),
    gridspec_kw={"height_ratios": [1, 3], "width_ratios": [20, 1]},
)
plt.subplots_adjust(hspace=0.05)
axs[0, 0].set_xscale("log")
axs[1, 0].set_xlabel(r"$r_0$ [mm]")
axs[1, 0].set_ylabel("surface RH [ ]")
axs[0, 0].tick_params(right=True, which="both")
axs[1, 0].tick_params(right=True, which="both")
axs[1, 0].tick_params(top=True, which="both")
axs[0, 0].tick_params(top=True, which="both")
axs[0, 0].set_ylabel("fraction mass \n evaporated [ ]")
axs[0, 0].set_xlim(r0grid[0, 0] * 1e3, r0grid[0, -1] * 1e3)
axs[0, 0].set_ylim(-0.04, 1.04)
levels_smooth = np.linspace(0, 1, 250)
cmesh = axs[1, 0].contourf(
    r0grid * 1e3,
    RHgrid,
    m_frac_evap,
    cmap=plt.cm.binary,
    vmin=0,
    vmax=1,
    levels=levels_smooth,
)


cb = f.colorbar(cmesh, cax=axs[1, 1])
cb.solids.set_edgecolor("face")
axs[0, 1].axis("off")
cb.solids.set_edgecolor("face")
cb.solids.set_linewidth(1e-5)
cb.set_label("fraction mass evaporated [ ]")
cb.set_ticks([0, 0.1, 0.25, 0.5, 0.75, 1])
axs[1, 0].axhline(0.75, lw=0.5, ls="--", c="plum")
axs[1, 0].plot(r_mins * 1e3, RHs, lw=3, c="darkviolet", zorder=10)
c_10 = axs[1, 0].contour(
    r0grid * 1e3,
    RHgrid,
    m_frac_evap,
    colors="indigo",
    linewidths=1,
    linestyles="--",
    levels=[0.1],
)
cb.add_lines(c_10)
axs[1, 0].clabel(
    c_10, c_10.levels, fmt={0.1: "10% mass evaporated"}, fontsize="smaller"
)
axs[1, 0].fill_betweenx(
    RHs, 1e-2, (r_mins - 1e-6) * 1e3, edgecolor="k", facecolor="w", hatch="//"
)
axs[1, 0].annotate(
    text="TOTAL \\n EVAPORATION", xy=(0.04, 0.55), c="k", backgroundcolor="w"
)
axs[1, 0].annotate(
    text=r"$r_\mathrm{min}$",
    xy=(0.35, 0.27),
    c="darkviolet",
    backgroundcolor="w",
    size=8,
)

axs[0, 0].scatter(r_mins[i_RH75] * 1e3, 0.99, color="darkviolet", zorder=10)
axs[0, 0].axvline(r_mins[i_RH75] * 1e3, lw=0.5, c="darkviolet", ls="--")
axs[0, 0].plot(r0grid[0, :] * 1e3, m_frac_evap[i_RH75, :], lw=2.05, c="k")
axs[0, 0].axhline(1, c="w", lw=3)
axs[0, 0].plot([1e-2, r_mins[i_RH75] * 1e3], [1, 1], c="k", lw=2.05, ls="--")
axs[0, 0].annotate(text="surface RH = 0.75", xy=(0.8, 0.85), c="plum", size=8)
axs[0, 0].annotate(text=r"$r_\mathrm{min}$", xy=(0.13, 0.05), c="darkviolet", size=8)

figs_path = os.path.join(dir, "figs")
os.mkdir(figs_path) if not os.path.exists(figs_path) else None
plt.savefig(
    os.path.join(figs_path, "fig02.pdf"),
    transparent=True,
    bbox_inches="tight",
    pad_inches=0.5,
)
plt.close()

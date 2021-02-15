from PySDM.physics.constants import si, convert_to
from matplotlib import pyplot
from PySDM_examples.utils.show_plot import show_plot
import numpy as np


def plot(var, qlabel, fname, output):
    dt = output['t'][1] - output['t'][0]
    dz = output['z'][1] - output['z'][0]
    tgrid = np.concatenate(((output['t'][0] - dt / 2,), output['t'] + dt / 2))
    zgrid = np.concatenate(((output['z'][0] - dz / 2,), output['z'] + dz / 2))
    convert_to(zgrid, si.km)

    fig = pyplot.figure(constrained_layout=True)
    gs = fig.add_gridspec(25, 5)
    ax1 = fig.add_subplot(gs[:-1, 0:4])
    mesh = ax1.pcolormesh(tgrid, zgrid, output[var], cmap='twilight')

    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('z [km]')

    cbar = fig.colorbar(mesh, fraction=.05, location='top')
    cbar.set_label(qlabel)

    ax2 = fig.add_subplot(gs[:-1, 4:], sharey=ax1)
    ax2.set_xlabel(qlabel)
    #ax2.set_yticks([], minor=False)

    last_t = 0
    for i, t in enumerate(output['t']):
        x, z = output[var][:, i], output['z'].copy()
        convert_to(z, si.km)
        params = {'color': 'black'}
        for line_t, line_s in {10: ':', 15: '--', 20: '-', 25: '-.'}.items():
            if last_t < line_t * si.min <= t:
                params['ls'] = line_s
                ax2.plot(x, z, **params)
                ax1.axvline(t, **params)
        last_t = t

    show_plot(filename=fname)

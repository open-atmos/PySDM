from matplotlib import pyplot
import matplotlib
import numpy as np


def _transform(data):
    return data.T


def image(ax, data, domain_size_in_metres, label, cmap='YlGnBu', scale='linear'):
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    im = ax.imshow(_transform(data),
                   origin='lower',
                   extent=(0, domain_size_in_metres[0], 0, domain_size_in_metres[1]),
                   cmap=cmap,
                   norm=matplotlib.colors.LogNorm() if scale == 'log' and np.isfinite(data).all() else None
                   )
    pyplot.colorbar(im, ax=ax).set_label(label)
    return im, ax


def image_update(im, ax, data):
    im.set_data(_transform(data))
    ax.set_title(f"min:{np.amin(data):.4g}    max:{np.amax(data):.4g}    std:{np.std(data):.4g}")


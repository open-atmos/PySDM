from matplotlib import pyplot


def _transform(data):
    return data.T


def image(ax, data, domain_size_in_metres, label, cmap='YlGnBu'):
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    im = ax.imshow(_transform(data),
                   origin='lower',
                   extent=(0, domain_size_in_metres[0], 0, domain_size_in_metres[1]),
                   cmap=cmap
                   )
    pyplot.colorbar(im, ax=ax).set_label(label)
    return im


def image_update(im, data):
    im.set_data(_transform(data))

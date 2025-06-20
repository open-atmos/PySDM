from PySDM import Formulae
from PySDM.physics import si
from PySDM.initialisation.sampling.spectral_sampling import AlphaSampling
from PySDM.initialisation import spectra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

formulae = Formulae()
n_sd = 2**13
mode1 = spectra.Lognormal(
                norm_factor=90 / si.centimetre**3,
            m_mode=0.03 * si.micrometre,
            s_geom=1.28,
        )
mode2 = spectra.Lognormal(
            norm_factor=15 / si.centimetre**3,
            m_mode=0.14 * si.micrometre,
            s_geom=1.75,
        )
spectrum = spectra.Sum((mode1, mode2))

fig,axs = plt.subplots(3, 1, figsize=(6, 12))
alphas = np.linspace(0,1,11)
for alpha in alphas:
    sampling = AlphaSampling(spectrum, alpha=alpha)
    xa,ya = sampling.sample(n_sd)
    sns.kdeplot(xa, ax=axs[0], label=f'alpha={alpha:.2f}',common_norm=True)
    axs[1].plot(xa, ya, label=f'alpha={alpha:.2f}')
    axs[2].plot(np.sort(ya),label=f'alpha={alpha:.2f}')

plt.setp(axs[0], 
         title='SD PDF', 
         xscale='log',
         xlim=(1e-8, 1e-6), ylim=(3e5, 0.4e8))
plt.setp(axs[1], ylabel='Multiplicity', xscale='log', yscale='log', xlim=(1e-8, 1e-6), ylim=(1e-3, 1e7))
plt.setp(axs[2], ylabel='Sorted Multiplicity', yscale='log', xlim=(0, n_sd), ylim=(1e-3, 1e7))
"""
box-model and 2D prescribed-flow immersion-freezing examples based on
[Arabas et al. 2025](https://doi.org/10.1029/2024MS004770)

aida.ipynb:
.. include:: ./aida.ipynb.badges.md

copula_hello.ipynb:
.. include:: ./copula_hello.ipynb.badges.md

fig_2.ipynb:
.. include:: ./fig_2.ipynb.badges.md

figs_10_and_11_and_animations.ipynb:
.. include:: ./figs_10_and_11_and_animations.ipynb.badges.md

fig_A2.ipynb:
.. include:: ./fig_A2.ipynb.badges.md

figs_3_and_7_and_8.ipynb:
.. include:: ./figs_3_and_7_and_8.ipynb.badges.md

figs_5_and_6.ipynb:
.. include:: ./figs_5_and_6.ipynb.badges.md
"""

from .make_particulator import make_particulator
from .plots import make_freezing_spec_plot, make_pdf_plot, make_temperature_plot
from .run_simulation import run_simulation

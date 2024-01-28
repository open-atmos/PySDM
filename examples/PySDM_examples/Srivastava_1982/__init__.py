"""
box-model examples feat. analytic solution for breakup from
[Srivastava 1982 (JAS)](https://doi.org/10.1175/1520-0469(1982)039%3C1317:ASMOPC%3E2.0.CO;2)
"""

from .equations import Equations, EquationsHelpers
from .example import (
    add_to_plot_simulation_results,
    coalescence_and_breakup_eq13,
    compute_log_space,
    get_coalescence_analytic_results,
    get_processed_results,
    get_pysdm_secondary_products,
)
from .settings import Settings, SimProducts
from .simulation import Simulation

from PySDM_examples.Kreidenweis_et_al_2003 import Settings, Simulation
from PySDM.physics import si


def test_cloud_base():
    # Arrange
    settings = Settings(n_sd=100, dt=1*si.s)
    simulation = Simulation(settings)

    # Act
    simulation.run(steps=1)

    # Assert
    simulation.core.products['RH'].get()

    # TODO (check what can be checked against Table 3)
    # Relative humidity at t = 0 	95 (%)
    # Cloud base height above surface 	698 (m)
    # Cloud base pressure 	939 (mbar)
    # Cloud base temperature 	284.2 (K)
    # Air density at cloud base 	1.15 (kg m−3)
    # Cloud water mixing ratio 1200 m (2400 s) above cloud base 	2.17 (g kg−1)
    # Chemical Conditions
    # SO2 at t = 0 	200 (ppt‐v)
    # NH3(g) at t = 0 	100 (ppt‐v)
    # H2O2 at t = 0 	500 (ppt‐v)
    # HNO3 at t = 0 	100 (ppt‐v)
    # O3 at t = 0 	50 (ppb‐v)
    # CO2 at t = 0 	360 (ppm‐v)
    # SO4= (particulate) at t = 0 	2 (μg m−3)
    # NH4+ (particulate) at t = 0 	0.375 (μg m−3)

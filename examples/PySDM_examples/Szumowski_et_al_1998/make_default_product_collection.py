import numpy as np

from PySDM import products as PySDM_products


def make_default_product_collection(settings):
    cloud_range = (settings.aerosol_radius_threshold, settings.drizzle_radius_threshold)
    products = [
        # Note: consider better radius_bins_edges
        PySDM_products.ParticleSizeSpectrumPerMassOfDryAir(
            name="Particles Wet Size Spectrum",
            unit="mg^-1 um^-1",
            radius_bins_edges=settings.r_bins_edges,
        ),
        PySDM_products.ParticleSizeSpectrumPerMassOfDryAir(
            name="Particles Dry Size Spectrum",
            unit="mg^-1 um^-1",
            radius_bins_edges=settings.r_bins_edges,
            dry=True,
        ),
        PySDM_products.TotalParticleConcentration(),
        PySDM_products.TotalParticleSpecificConcentration(),
        PySDM_products.ParticleConcentration(
            radius_range=(0, settings.aerosol_radius_threshold)
        ),
        PySDM_products.ParticleConcentration(
            name="n_c_cm3", unit="cm^-3", radius_range=cloud_range
        ),
        PySDM_products.WaterMixingRatio(
            name="cloud water mixing ratio", radius_range=cloud_range
        ),
        PySDM_products.WaterMixingRatio(
            name="rain water mixing ratio",
            radius_range=(settings.drizzle_radius_threshold, np.inf),
        ),
        PySDM_products.ParticleConcentration(
            name="drizzle concentration",
            radius_range=(settings.drizzle_radius_threshold, np.inf),
            unit="cm^-3",
        ),
        PySDM_products.ParticleSpecificConcentration(
            name="aerosol specific concentration",
            radius_range=(0, settings.aerosol_radius_threshold),
            unit="mg^-1",
        ),
        PySDM_products.MeanRadius(unit="um"),
        PySDM_products.SuperDropletCountPerGridbox(),
        PySDM_products.AmbientRelativeHumidity(name="RH_env", var="RH"),
        PySDM_products.AmbientPressure(name="p_env", var="p"),
        PySDM_products.AmbientTemperature(name="T_env", var="T"),
        PySDM_products.AmbientWaterVapourMixingRatio(
            name="water_vapour_mixing_ratio_env", var="water_vapour_mixing_ratio"
        ),
        PySDM_products.AmbientDryAirDensity(name="rhod_env", var="rhod"),
        PySDM_products.AmbientDryAirPotentialTemperature(name="thd_env", var="thd"),
        PySDM_products.CPUTime(),
        PySDM_products.WallTime(),
        PySDM_products.EffectiveRadius(unit="um", radius_range=cloud_range),
        PySDM_products.RadiusBinnedNumberAveragedTerminalVelocity(
            radius_bin_edges=settings.terminal_velocity_radius_bin_edges
        ),
    ]

    if settings.processes["fluid advection"]:
        products.append(PySDM_products.MaxCourantNumber())
        products.append(PySDM_products.CoolingRate(unit="K/min"))
    if settings.processes["condensation"]:
        products.append(PySDM_products.CondensationTimestepMin(name="dt_cond_min"))
        products.append(PySDM_products.CondensationTimestepMax(name="dt_cond_max"))
        products.append(PySDM_products.PeakSupersaturation(unit="%", name="S_max"))
        products.append(PySDM_products.ActivatingRate())
        products.append(PySDM_products.DeactivatingRate())
        products.append(PySDM_products.RipeningRate())
    if settings.processes["particle advection"]:
        products.append(
            PySDM_products.SurfacePrecipitation(name="surf_precip", unit="mm/day")
        )
    if settings.processes["coalescence"]:
        products.append(PySDM_products.CollisionTimestepMean(name="dt_coal_avg"))
        products.append(PySDM_products.CollisionTimestepMin(name="dt_coal_min"))
        products.append(PySDM_products.CollisionRatePerGridbox(name="cr"))
        products.append(PySDM_products.CollisionRateDeficitPerGridbox(name="crd"))
        products.append(PySDM_products.CoalescenceRatePerGridbox(name="cor"))
    if settings.processes["breakup"]:
        products.append(PySDM_products.BreakupRatePerGridbox(name="br"))
        products.append(PySDM_products.BreakupRateDeficitPerGridbox(name="brd"))
    if settings.processes["freezing"]:
        products.append(PySDM_products.IceWaterContent())
        if settings.freezing_singular:
            products.append(
                PySDM_products.FreezableSpecificConcentration(settings.T_bins_edges)
            )
        else:
            products.append(PySDM_products.TotalUnfrozenImmersedSurfaceArea())
            # TODO #599 immersed surf spec
        products.append(
            PySDM_products.ParticleSpecificConcentration(
                radius_range=(-np.inf, 0), name="n_ice"
            )
        )

    return products

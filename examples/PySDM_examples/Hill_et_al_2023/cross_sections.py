from PySDM.physics import si
import numpy as np


def compute_dvol_andsigma_time_series(output, settings, cloud_base):
    num_conc = (output["nc"] + output["nr"])[cloud_base]
    rho_w = settings.formulae.constants.rho_w
    D_vol = (6 * output["LWC"][cloud_base] / (np.pi * num_conc * rho_w)) ** (1 / 3)

    r_bin_edges = settings.r_bins_edges
    r_bin_centers = (r_bin_edges[:-1] + r_bin_edges[1:]) / 2
    r_bin_volumes = settings.formulae.trivia.volume(r_bin_centers)

    nc = output["wet spectrum"][cloud_base] + output["dry spectrum"][cloud_base]
    mass_bins = (nc * r_bin_volumes[:, np.newaxis]) * rho_w
    Dk = r_bin_centers * 2
    Dvolmass = np.sum(mass_bins * Dk[:, np.newaxis], axis=0) / np.sum(mass_bins, axis=0)
    Nd = np.sum(nc)
    sigma = np.sqrt(
        (1 / (Nd - 1)) * np.sum(nc * (Dk[:, np.newaxis] - Dvolmass) ** 2, axis=0)
    )
    return D_vol, sigma


def compute_dvol_andsigma_vertical(output, settings, zslice, time):
    t_idx = np.where(output["t"] == time)[0][0]
    num_conc = (output["nc"] + output["nr"])[zslice, t_idx]
    rho_w = settings.formulae.constants.rho_w

    D_vol = (6 * output["LWC"][zslice, t_idx] / (np.pi * num_conc * rho_w)) ** (1 / 3)

    r_bin_edges = settings.r_bins_edges
    r_bin_centers = (r_bin_edges[:-1] + r_bin_edges[1:]) / 2
    r_bin_volumes = settings.formulae.trivia.volume(r_bin_centers)
    nc = (
        output["wet spectrum"][zslice, :, t_idx]
        + output["dry spectrum"][zslice, :, t_idx]
    )
    mass_bins = (nc * r_bin_volumes) * rho_w
    Dk = r_bin_centers * 2
    Dvolmass = np.sum(mass_bins * Dk, axis=1) / np.sum(mass_bins, axis=1)
    Nd = np.sum(nc, axis=1)
    sigma = np.sqrt(
        (1 / (Nd - 1)) * np.sum(nc * (Dk - Dvolmass[:, np.newaxis]) ** 2, axis=1)
    )
    return D_vol, sigma


def compute_LWP_and_nc_time_series(output, settings, z_slice):
    LWP = np.sum(output["LWC"][z_slice], axis=0) * settings.dz
    num_conc = (output["nc"] + output["nr"])[z_slice]
    mask = output["LWC"][z_slice] > 1e-5
    masked = num_conc * mask
    masked[masked == 0] = np.nan
    mean_num_conc = np.nanmean(masked, axis=0)
    mean_num_conc[mean_num_conc == np.nan] = 0
    return LWP, mean_num_conc


def compute_LWP_and_nc_vertical(output, z_slice, time):
    t_idx = np.where(output["t"] == time)[0][0]
    LWC = output["LWC"][z_slice, t_idx]
    num_conc = (output["nc"] + output["nr"])[z_slice, t_idx]
    mask = output["LWC"][z_slice, t_idx] > 1e-5
    masked = num_conc * mask
    masked[masked == 0] = np.nan
    mean_num_conc = masked
    mean_num_conc[mean_num_conc == np.nan] = 0
    return LWC, num_conc

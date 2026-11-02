"""
reference results for bulk scheme in Fig B1. in
[Spichtinger et al. 2023](https://doi.org/10.5194/acp-23-2035-2023)
"""

import numpy as np


def bulk_model_reference_array():

    initial_temperatures = np.array([196.0, 216.0, 236.0])
    updrafts = np.array([0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0])

    dim_size = (np.shape(initial_temperatures)[0], np.shape(updrafts)[0])
    ni_bulk_ref = np.zeros(dim_size)

    # T = 196
    ni_bulk_ref[0, 0] = 643686.1316903427
    ni_bulk_ref[0, 1] = 2368481.0609527444
    ni_bulk_ref[0, 2] = 20160966.984670535
    ni_bulk_ref[0, 3] = 49475281.81718969
    ni_bulk_ref[0, 4] = 131080662.23620115
    ni_bulk_ref[0, 5] = 401046528.70428866
    ni_bulk_ref[0, 6] = 627442148.3402529
    ni_bulk_ref[0, 7] = 1151707310.2210448

    # T = 216
    ni_bulk_ref[1, 0] = 60955.84292640147
    ni_bulk_ref[1, 1] = 189002.0792186534
    ni_bulk_ref[1, 2] = 1200751.6897658105
    ni_bulk_ref[1, 3] = 2942110.815055958
    ni_bulk_ref[1, 4] = 10475282.894692907
    ni_bulk_ref[1, 5] = 90871045.40856971
    ni_bulk_ref[1, 6] = 252175505.460412
    ni_bulk_ref[1, 7] = 860335156.4717773

    # T = 236
    ni_bulk_ref[2, 0] = 13049.108886452004
    ni_bulk_ref[2, 1] = 40422.244759544985
    ni_bulk_ref[2, 2] = 237862.49854786208
    ni_bulk_ref[2, 3] = 545315.7805748513
    ni_bulk_ref[2, 4] = 1707801.469906006
    ni_bulk_ref[2, 5] = 11128055.66932415
    ni_bulk_ref[2, 6] = 27739585.111447476
    ni_bulk_ref[2, 7] = 101799566.47225031

    return initial_temperatures, updrafts, ni_bulk_ref

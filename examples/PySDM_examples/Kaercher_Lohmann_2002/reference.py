import numpy as np
from PySDM.physics.constants import si

def critical_supersaturation(temperature):
    return 2.349 - temperature / 259.

# crit thres
def critical_supersaturation_spich2023 (t):
    s20=1.67469
    s21=0.00228125
    s22=-1.36989e-05
    return s20+s21*t+s22*t*t


def bulk_model_reference(initial_temperature, updraft=0.1):
    n_hom_ice = None

    if initial_temperature == 220.:
        if updraft == 0.1:
            n_hom_ice = 148121.413358197
        if updraft == 1.:
            n_hom_ice = 7268664.77542974



    return( n_hom_ice / si.metre ** 3  )


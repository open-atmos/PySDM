# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM.physics import si
from PySDM_examples.Abdul_Razzak_Ghan_2000.run_ARG_parcel import run_parcel


def test_just_do_it():
    run_parcel(
        w=1 * si.m / si.s,
        sol2=.5,
        N2=100 / si.cm**3,
        rad2=100 * si.nm,
        n_sd_per_mode=10
    )

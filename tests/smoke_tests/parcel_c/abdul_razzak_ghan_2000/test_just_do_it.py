# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from PySDM_examples.Abdul_Razzak_Ghan_2000.run_ARG_parcel import run_parcel

from PySDM.physics import si


def test_just_do_it():
    # act
    output = run_parcel(
        w=1 * si.m / si.s,
        sol2=0.5,
        N2=100 / si.cm**3,
        rad2=100 * si.nm,
        n_sd_per_mode=10,
    )

    # assert
    assert (output.activated_fraction_S[:] <= 1).all()

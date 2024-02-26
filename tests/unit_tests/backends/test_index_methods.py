# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import matplotlib.pyplot as plt
import numpy as np
import pytest

from PySDM.backends.impl_numba.methods.index_methods import (
    IndexMethods,
    draw_random_int,
    fisher_yates_shuffle,
)


@pytest.mark.parametrize(
    "a, b, u01, expected",
    (
        (0, 100, 0.0, 0),
        (0, 100, 1.0, 100),
        (0, 1, 0.5, 1),
        (0, 1, 0.49, 0),
        (0, 3, 0.49, 1),
        (0, 3, 0.245, 0),
        (0, 2, 0.332, 0),
        (0, 2, 0.333, 0),
        (0, 2, 0.334, 1),
        (0, 2, 0.665, 1),
        (0, 2, 0.666, 1),
        (0, 2, 0.667, 2),
        (0, 2, 0.999, 2),
    ),
)
def test_draw_random_int(a, b, u01, expected):
    # act
    actual = draw_random_int(a, b, u01)

    # assert
    assert actual == expected


def test_fisher_yates_shuffle():
    # arrange
    n = 10

    idx = np.arange(n)
    random_nums = np.linspace(1, 0, n + 2)[1:-1]

    # act
    fisher_yates_shuffle(idx, random_nums, 0, len(idx))

    # assert
    expected = np.array([9, 8, 3, 4, 5, 6, 7, 2, 1, 0])
    assert np.all(expected == idx)


@pytest.mark.parametrize("seed", (1, 2, 3, 1231, 42))
# pylint: disable=redefined-outer-name
def test_shuffle_global_generates_uniform_distribution(seed, plot=False):
    # arrange
    np.random.seed(seed)

    n = 4
    possible_permutations_num = np.math.factorial(n)
    coverage = 1000

    n_runs = coverage * possible_permutations_num

    # act
    all_permutations = {}

    for _ in range(n_runs):
        idx = np.arange(0, n)
        u01 = np.random.uniform(0, 1, n)

        IndexMethods.shuffle_global(idx, len(idx), u01)
        # np.random.shuffle(idx)

        perm_tuple = (*idx,)
        if perm_tuple in all_permutations:
            all_permutations[perm_tuple] += 1
        else:
            all_permutations[perm_tuple] = 1

    all_permutations_vals = np.array(list(all_permutations.values()))
    deviations_from_mean = all_permutations_vals - coverage

    plt.bar(np.arange(possible_permutations_num), deviations_from_mean)
    plt.axhline(0, color="green")

    if plot:
        plt.show()

    # assert
    assert len(all_permutations.keys()) == possible_permutations_num
    assert np.amax(np.abs(deviations_from_mean)) < coverage * 0.15

    tmp = np.abs(all_permutations_vals - coverage) ** 2.0
    std = np.sqrt(np.mean(tmp))
    assert std < coverage * 0.05


def test_merge_shuffle_equals_fisher_yates_when_depth_0():
    pass  # TODO:

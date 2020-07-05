"""
Created at 21.03.2020
"""

import numpy as np
import warnings
from .__parametrisation__ import backend


def universal_test(method_name, sut, params):
    # Arrange
    default_params = {}
    sut_params = {}
    for param in params:
        if param['name'] == 'idx':
            generate_param = generate_idx
        elif param['name'] == 'length':
            generate_param = generate_length
        elif param['name'] == 'is_first_in_pair':
            generate_param = generate_is_first_in_pair
        else:
            generate_param = generate_data
        sut_params[param['name']], default_params[param['name']] = generate_param(sut, **param['details'])
    default_method = getattr(backend, method_name)
    sut_method = getattr(sut, method_name)

    # Act
    default_result = default_method(**default_params)
    sut_result = sut_method(**sut_params)

    # Assert
    precision = 13
    for param in params:
        if 'value' in param['details'].keys() or param['name'] == 'length':
            assert sut_params[param['name']] == default_params[param['name']]
        else:
            sut_param = sut.to_ndarray(sut_params[param['name']])
            default_param = backend.to_ndarray(default_params[param['name']])
            if 'checking' in param.keys():
                if 'length_valid' in param['checking']:
                    sut_param = sut_param[:default_params['length']]
                    default_param = default_param[:default_params['length']]
                if 'sorted' in param['checking']:
                    sut_param = sorted(sut_param)
                    default_param = sorted(default_param)
            try:
                np.testing.assert_array_equal(
                    sut_param,
                    default_param
                )
            except AssertionError:
                warnings.warn(f"{param['name']} fail with high precision, try with {precision} digit precision...")
                np.testing.assert_almost_equal(
                    sut_param,
                    default_param,
                    decimal=precision
                )
    if default_result is None:
        assert sut_result is None
    elif isinstance(default_result, (int, float, bool)):
        assert sut_result == default_result
    else:
        np.testing.assert_array_equal(sut.to_ndarray(sut_result), backend.to_ndarray(default_result))


def generate_data(sut_backend, shape=None, dtype=None, value=None, array=None, seed=0, negative=False, factor=100):
    if value is None and array is None:
        np.random.seed(seed)
        rand_ndarray = (factor * np.random.rand(*shape)).astype(dtype)

        if negative:
            rand_ndarray = (rand_ndarray - factor / 2) * 2

        result_sut = sut_backend.from_ndarray(rand_ndarray)
        result_default = backend.from_ndarray(rand_ndarray)
    elif array is None and shape is None and dtype is None:
        if isinstance(value, (float, int, bool)):
            result_sut = value
            result_default = value
        else:
            raise ValueError()
    elif value is None and shape is None and dtype is None:
        if isinstance(array, np.ndarray):
            result_sut = sut_backend.from_ndarray(array)
            result_default = backend.from_ndarray(array)
        else:
            raise ValueError()
    else:
        raise ValueError()

    return result_sut, result_default


def idx_length(shape):
    if len(shape) >= 2:
        result = shape[1]
    else:
        result = shape[0]

    return result


def generate_idx(sut_backend, shape, order='asc', seed=0):
    np.random.seed(seed)

    idx_len = idx_length(shape)

    idx_ndarray = np.arange(idx_len)

    if order == 'desc':
        idx_ndarray = idx_ndarray[::-1]
    elif order == 'random':
        np.random.permutation(idx_ndarray)

    result_sut = sut_backend.from_ndarray(idx_ndarray)
    result_default = backend.from_ndarray(idx_ndarray)

    return result_sut, result_default


def generate_length(_sut_backend, length, shape):
    idx_len = idx_length(shape)

    if length == 'zero':
        result = 0
    elif length == 'middle':
        result = (idx_len + 1) // 2
    elif length == 'full':
        result = idx_len
    return result, result


def generate_is_first_in_pair(_sut_backend, shape, pairs='random', seed=0):
    np.random.seed(seed)

    idx_len = idx_length(shape)

    if pairs == 'none':
        is_first_in_pair = np.zeros(idx_len, dtype=np.int64)
    elif pairs == 'random':
        is_first_in_pair = np.random.random_integers(low=0, high=1, size=idx_len)
        for i in range(idx_len-1):
            if is_first_in_pair[i] == 1:
                is_first_in_pair[i + 1] = 0
        is_first_in_pair[-1] = 0
    elif pairs == 'full':
        is_first_in_pair = np.zeros(idx_len, dtype=np.int64)
        is_first_in_pair[:-1:2] = 1

    result_sut = _sut_backend.from_ndarray(is_first_in_pair)
    result_default = backend.from_ndarray(is_first_in_pair)

    return result_sut, result_default

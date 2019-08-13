"""
Created at 09.08.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


# pythran export amin(float64[:], int64[:], int)
# pythran export amin(int64[:], int64[:], int)
def amin(row, idx, length):
    result = np.amin(row[idx[:length]])
    return result


# pythran export amax(float64[:], int64[:], int)
# pythran export amax(int64[:], int64[:], int)
def amax(row, idx, length):
    result = np.amax(row[idx[:length]])
    return result


# pythran export remove_zeros(int64[:], int64[:], int)
def remove_zeros(data, idx, length):
    result = 0
    for i in range(length):
        if data[idx[i]] == 0:
            idx[i] = len(idx)
        else:
            result += 1
    idx[:length] = np.sort(idx[:length])
    return result


# pythran export coalescence(int64[:], int64[:], int, float64[:, :], float64[:, :], float64[:], int64[:])
def coalescence(n, idx, length, intensive, extensive, gamma, healthy):
    # omp parallel for
    for i in range(length // 2):
        j = 2 * i
        k = j + 1

        j = idx[j]
        k = idx[k]

        if n[j] < n[k]:
            j, k = k, j
        g = min(gamma[i], n[j] // n[k])
        if g == 0:
            continue

        new_n = n[j] - g * n[k]
        if new_n > 0:
            n[j] = new_n
            extensive[:, k] += g * extensive[:, j]
        else:  # new_n == 0
            n[j] = n[k] // 2
            n[k] = n[k] - n[j]
            extensive[:, j] = g * extensive[:, j] + extensive[:, k]
            extensive[:, k] = extensive[:, j]
        if n[k] == 0 or n[j] == 0:
            healthy[0] = 0


# pythran export sum_pair(float64[:], float64[:], int64[:], int)
def sum_pair(data_out, data_in, idx, length):
    # omp parallel for
    for i in range(length // 2):
        data_out[i] = data_in[idx[2 * i]] + data_in[idx[2 * i + 1]]


# pythran export max_pair(float64[:], int64[:], int64[:], int)
def max_pair(data_out, data_in, idx, length):
    # omp parallel for
    for i in range(length // 2):
        data_out[i] = max(data_in[idx[2 * i]], data_in[idx[2 * i + 1]])


# pythran export multiply(float64[:], float)
# pythran export multiply(float64[:], float64[:])
def multiply(data, multiplier):
    data *= multiplier


# pythran export sum(float64[:], float64[:])
def sum(data_out, data_in):
    data_out[:] = data_out + data_in


# pythran export floor(float[])
def floor(row):
    row[:] = np.floor(row)


# pythran export first_element_is_zero(int64[:])
def first_element_is_zero(arr):
    return arr[0] == 0
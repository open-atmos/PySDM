# adapted from Maciej Waruszewski's libcloudph++ version of Boost implementation
# https://github.com/igfuw/libcloudphxx/blob/master/include/libcloudph%2B%2B/common/detail/toms748.hpp
# Boost version: (C) Copyright John Maddock 2006 (http://www.boost.org/LICENSE_1_0.txt)

import numba
from sys import float_info
from PySDM.backends.numba.conf import JIT_FLAGS
from numpy import nan

float_info_epsilon = float_info.epsilon
float_info_max = float_info.max
float_info_min = float_info.min


@numba.njit(**{**JIT_FLAGS, **{'parallel': False, 'cache':False}})
def bracket(f, args, a, b, c, fa, fb):
    tol = float_info_epsilon * 2
    if (b - a) < 2 * tol * a:
        c = a + (b - a) / 2
    elif c <= a + abs(a) * tol:
        c = a + abs(a) * tol
    elif c >= b - abs(b) * tol:
        c = b - abs(a) * tol
    fc = f(c, *args)
    if fc == 0:
        a = c
        fa = 0
        d = 0
        fd = 0
    else:
        if fa * fc < 0:
            d = b
            fd = fb
            b = c
            fb = fc
        else:
            d = a
            fd = fa
            a = c
            fa= fc
    return a, b, fa, fb, d, fd


@numba.njit(**{**JIT_FLAGS, **{'parallel': False}})
def safe_div(num, denom, r):
    if abs(denom) < 1:
        if abs(denom * float_info_max) <= abs(num):
            return r
    return num / denom


@numba.njit(**{**JIT_FLAGS, **{'parallel': False}})
def secant_interpolate(a, b, fa, fb):
    tol = float_info_epsilon * 5
    c = a - (fa / (fb - fa)) * (b - a)
    if c <= a + abs(a) * tol or c >= b - abs(b) * tol:
        return (a + b) / 2
    return c


@numba.njit(**{**JIT_FLAGS, **{'parallel': False}})
def quadratic_interpolate(a, b, d, fa, fb, fd, count):
    B = safe_div(fb - fa, b - a, float_info_max)
    A = safe_div(fd - fb, d - b, float_info_max)
    A = safe_div(A - B, d - a, 0.)

    if A == 0:
        return secant_interpolate(a, b, fa, fb)
    if A * fa > 0:
        c = a
    else:
        c = b
    for i in range(1, count + 1):
        c -= safe_div(
            fa + (B + A * (c - b)) * (c - a),
            B + A * (2. * c - a - b),
            1. + c - a
        )
    if (c <= a) or (c >= b):
        c = secant_interpolate(a, b, fa, fb)
    return c


@numba.njit(**{**JIT_FLAGS, **{'parallel': False}})
def cubic_interpolate(a, b, d, e, fa, fb, fd, fe):
    q11 = (d - e) * fd / (fe - fd)
    q21 = (b - d) * fb / (fd - fb)
    q31 = (a - b) * fa / (fb - fa)
    d21 = (b - d) * fd / (fd - fb)
    d31 = (a - b) * fb / (fb - fa)

    q22 = (d21 - q11) * fb / (fe - fb)
    q32 = (d31 - q21) * fa / (fd - fa)
    d32 = (d31 - q21) * fd / (fd - fa)
    q33 = (d32 - q22) * fa / (fe - fa)
    c = q31 + q32 + q33 + a

    if (c <= a) or (c >= b):
        c = quadratic_interpolate(a, b, d, fa, fb, fd, 3)
    return c


@numba.njit(**{**JIT_FLAGS, **{'parallel': False}})
def tol(a, b, rtol, within_tolerance):
    return within_tolerance(abs(a - b), min(abs(a), abs(b)), rtol)


@numba.njit(**{**JIT_FLAGS, **{'parallel': False}})
def toms748_solve(f, args, ax, bx, fax, fbx, rtol, max_iter, within_tolerance):
    count = max_iter
    mu = 0.5
    a = ax
    b = bx
    fa = fax
    fb = fbx
    if not a < b:
        print("TOMS748 problem: not a < b")
        return nan, -1

    if tol(a, b, rtol, within_tolerance) or fa == 0 or fb == 0:
        max_iter = 0
        if fa == 0:
            b = a
        elif fb == 0:
            a = b
        return (a + b) / 2, max_iter

    if not fa * fb < 0:
        print("TOMS748 problem: not fa * fb < 0")
        return nan, -1

    fe = e = fd = 1e5

    if fa != 0:
        c = secant_interpolate(a, b, fa, fb)
        a, b, fa, fb, d, fd = bracket(f, args, a, b, c, fa, fb)
        count -= 1

        if count > 0 and fa != 0 and not tol(a, b, rtol, within_tolerance):
            c = quadratic_interpolate(a, b, d, fa, fb, fd, 2)
            e = d
            fe = fd
            a, b, fa, fb, d, fd = bracket(f, args, a, b, c, fa, fb)
            count -= 1

    while count > 0 and fa != 0 and not tol(a, b, rtol, within_tolerance):
        a0 = a
        b0 = b
        min_diff = float_info_min * 32
        prof = (
            abs(fa - fb) < min_diff or
            abs(fa - fd) < min_diff or
            abs(fa - fe) < min_diff or
            abs(fb - fd) < min_diff or
            abs(fb - fe) < min_diff or
            abs(fd - fe) < min_diff
        )
        if prof:
            c = quadratic_interpolate(a, b, d, fa, fb, fd, 2)
        else:
            c = cubic_interpolate(a, b, d, e, fa, fb, fd, fe)
        e = d
        fe = fd
        a, b, fa, fb, d, fd = bracket(f, args, a, b, c, fa, fb)
        if 1 == count or fa == 0 or tol(a, b, rtol, within_tolerance):
            count -= 1
            break
        prof = (
            abs(fa - fb) < min_diff or
            abs(fa - fd) < min_diff or
            abs(fa - fe) < min_diff or
            abs(fb - fd) < min_diff or
            abs(fb - fe) < min_diff or
            abs(fd - fe) < min_diff
        )
        if prof:
            c = quadratic_interpolate(a, b, d, fa, fb, fd, 3)
        else:
            c = cubic_interpolate(a, b, d, e, fa, fb, fd, fe)
        a, b, fa, fb, d, fd = bracket(f, args, a, b, c, fa, fb)
        if 1 == count or fa == 0 or tol(a, b, rtol, within_tolerance):
            count -= 1
            break
        if abs(fa) < abs(fb):
            u = a
            fu = fa
        else:
            u = b
            fu = fb
        c = u - 2 * (fu / (fb - fa)) * (b - a)
        if abs(c - u) > (b - a) / 2:
            c = a + (b - a) / 2
        e = d
        fe = fd
        a, b, fa, fb, d, fd = bracket(f, args, a, b, c, fa, fb)
        if 1 == count or fa == 0 or tol(a, b, rtol, within_tolerance):
            count -= 1
            break
        if (b - a) < mu * (b0 - a0):
            continue
        e = d
        fe = fd
        a, b, fa, fb, d, fd = bracket(f, args, a, b, a + (b - a) / 2, fa, fb)
        count -= 1

    max_iter -= count
    if fa == 0:
        b = a
    elif fb == 0:
        a = b
    return (a + b) / 2, max_iter

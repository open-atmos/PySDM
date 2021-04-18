from PySDM.backends.numba import conf

from PySDM.physics.impl import flag

if flag.DIMENSIONAL_ANALYSIS:
    from PySDM.physics.impl.fake_numba import njit
    formula = njit
else:
    import numba
    def formula(func=None, **kw):
        if func is None:
            return numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'inline': 'always', **kw}})
        else:
            return numba.njit(func, **{**conf.JIT_FLAGS, **{'parallel': False,  'inline': 'always', **kw}})

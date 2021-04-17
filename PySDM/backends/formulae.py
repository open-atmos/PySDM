from PySDM.physics import constants as const
from PySDM.backends.numba import conf
import inspect
import re
from PySDM.backends.thrustRTC.impl.precision_resolver import PrecisionResolver

from PySDM.physics import _flag
if _flag.DIMENSIONAL_ANALYSIS:
    from PySDM.physics._fake_numba import njit
    formula = njit
else:
    import numba
    def formula(func=None, **kw):
        if func is None:
            return numba.njit(**{**conf.JIT_FLAGS, **{'parallel': False, 'inline': 'always', **kw}})
        else:
            return numba.njit(func, **{**conf.JIT_FLAGS, **{'parallel': False,  'inline': 'always', **kw}})


def c_inline(fun, **args):
    prae = r"([+\-*/( ]|^)"
    post = r"([ )/*\-+]|$)"
    real_t = PrecisionResolver.get_C_type()
    source = inspect.getsourcelines(fun)[0]
    assert len(source) == 3
    source = source[-1].strip()
    source = re.sub("^return ", "", source)
    for arg in inspect.signature(fun).parameters:
        source = re.sub(f"{prae}({arg}){post}", f"\\1{real_t}({args[arg]})\\3", source)
    source = re.sub(f"{prae}const\\.([^\\d\\W]\\w*]*){post}", "\\1" + real_t + "({const.\\2})\\3", source)
    source = eval(f'f"""{source}"""')
    return f'{real_t}({source})'

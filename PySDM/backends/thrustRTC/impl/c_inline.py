import inspect
import re

from PySDM.backends.thrustRTC.impl.precision_resolver import PrecisionResolver
from PySDM.physics import constants as const

def c_inline(fun, **args):
    prae = r"([+\-*/( ]|^)"
    post = r"([ )/*\-+]|$)"
    real_t = PrecisionResolver.get_C_type()
    real_fmt = ".32g"
    source = inspect.getsourcelines(fun)[0]
    assert len(source) == 3
    source = source[-1].strip()
    source = re.sub("^return ", "", source)
    for arg in inspect.signature(fun).parameters:
        source = re.sub(f"{prae}({arg}){post}", f"\\1{real_t}({args[arg]})\\3", source)
    source = re.sub(
        f"{prae}const\\.([^\\d\\W]\\w*]*){post}",
        "\\1" + real_t + "({const.\\2:" + real_fmt + "})\\3",
        source
    )
    source = eval(f'f"""{source}"""')
    return f'{real_t}({source})'
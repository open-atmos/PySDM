"""
Created at 28.09.2020
"""

from ...numba.conf import JIT_FLAGS

cppython = {
    "int ": "",
    "double ": "",
    "float ": "",
    "auto ": "",
    "bool ": "",
    " {": ":",
    "}": "",
    "//": "#",
    "||": "or",
    "&&": "and",
    "(long)": "",
    "floor": "np.floor",
    "ceil": "np.ceil",
    "return": "continue",
}


def extract_var(cpp, start_position, end_char):
    stop = cpp.find(end_char, start_position)
    return cpp[start_position + 1: stop].strip()


def for_to_python(cpp: str):
    """
    need ';' in code and be running after removing types

    :param cpp:
    :return python code:
    """
    start = cpp.find("(")
    var = extract_var(cpp, start, "=")

    start = cpp.find("=")
    range_start = extract_var(cpp, start, ";")

    start = cpp.find("<")
    if start == -1:
        start = cpp.find(">")
    sign = "-" if cpp[start] == ">" else ""
    range_stop = extract_var(cpp, start, ";")

    start = cpp.find("+=")
    if start == -1:
        start = cpp.find("-=")
    start += 1

    range_step = extract_var(cpp, start, ")")

    return f"for {var} in range({range_start}, {range_stop}, {sign}{range_step}):"


def replace_fors(cpp):
    start = cpp.find("for ")
    while start > -1:
        stop = cpp.find(":", start)
        cpp_for = cpp[start:stop+1]
        python_for = for_to_python(cpp_for)
        cpp = cpp.replace(cpp_for, python_for.replace("for", "__python_token__"))
        start = cpp.find("for ", start + len(python_for))
    return cpp.replace("__python_token__", "for")

# TODO: wouldn't error_model='python' and fastmath=False be better for testing?
def to_numba(name, args, body):
    body = body.replace("\n", "\n    ")
    for cpp, python in cppython.items():
        body = body.replace(cpp, python)
    body = replace_fors(body)
    result = f'''
def make(self):
    import numpy as np
    import numba
    @numba.njit(parallel={JIT_FLAGS['parallel']}, error_model='numpy', fastmath=True)
    def {name}(__python_n__, {str(args).replace("'", "").replace('"', '')[1:-1]}):
        for i in numba.prange(__python_n__):
            {body}

    return {name}
'''

    return result

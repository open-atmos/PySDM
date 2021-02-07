"""
Created at 28.09.2020
"""

from ...numba.conf import JIT_FLAGS
import re
import numpy as np

cppython = {
    "int ": "",
    "double ": "",
    "float ": "",
    "auto ": "",
    "bool ": "",
    "ULLONG_MAX": f"{np.uint32(-1)}",  # note: would need more clever parsing to work with np.uint64(-1)
    " {": ":",
    "}": "",
    "//": "#",
    "||": "or",
    "&&": "and",
    "(int64_t)": "np.int64",  # TODO #324 unit test depicting what fails when this is changed to int16
    "(double)": "np.float64",
    "(float)": "np.float32",
    "floor": "np.floor",
    "ceil": "np.ceil",
    "return": "continue",
}


def extract_var(cpp, start_position, end_char):
    stop = cpp.find(end_char, start_position)
    return cpp[start_position + 1: stop].strip()


def for_to_python(cpp: str) -> str:
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


def replace_fors(cpp) -> str:
    start = cpp.find("for ")
    while start > -1:
        stop = cpp.find(":", start)
        cpp_for = cpp[start:stop + 1]
        python_for = for_to_python(cpp_for)
        cpp = cpp.replace(cpp_for, python_for.replace("for", "__python_token__"))
        start = cpp.find("for ", start + len(python_for))
    return cpp.replace("__python_token__", "for")


def atomic_min_to_python(cpp: str) -> str:
    cpp = cpp.replace("unsigned long long int*", "") \
              .replace("unsigned long long int", "np.uint64") \
              .replace("double*", "") \
              .replace("float*", "") \
              .replace(" ", "") \
              .replace("()", "") \
              .replace("&", "")
    cpp = re.sub(
        r"atomicMin\((\w+)\[(\w+)],\s*\(np.uint64\)\(([^)]*)\)\);",
        r"np.minimum(\1[\2:\2+1], np.asarray(\3, dtype=np.uint64), \1[\2:\2+1]);", cpp)
    print(cpp)
    return cpp


def replace_atomic_mins(cpp: str) -> (str, bool):
    start = cpp.find("atomicMin")
    parallel = start == -1
    while start > -1:
        stop = cpp.find(";", start)
        cpp_atomic_min = cpp[start:stop + 1]
        python_atomic_min = atomic_min_to_python(cpp_atomic_min)
        cpp = cpp.replace(cpp_atomic_min, python_atomic_min)
        start = cpp.find("atomicMin", start + len(python_atomic_min))
    return cpp, parallel


def atomic_add_to_python(cpp: str) -> str:
    cpp = cpp.replace("atomicAdd", "") \
              .replace("unsigned long long int*", "") \
              .replace("unsigned long long int", "") \
              .replace("double*", "") \
              .replace("float*", "") \
              .replace(" ", "") \
              .replace("()", "") \
              .replace("&", "") \
              .replace(",", "+=", 1) \
              [1:-2]  # remove '(' and ');'
    return cpp


def replace_atomic_adds(cpp: str) -> (str, bool):
    start = cpp.find("atomicAdd")
    parallel = start == -1
    while start > -1:
        stop = cpp.find(";", start)
        cpp_atomic_add = cpp[start:stop + 1]
        python_atomic_add = atomic_add_to_python(cpp_atomic_add)
        cpp = cpp.replace(cpp_atomic_add, python_atomic_add)
        start = cpp.find("atomicAdd", start + len(python_atomic_add))
    return cpp, parallel


def to_numba(name, args, iter_var, body):
    body = body.replace("\n", "\n    ")
    for cpp, python in cppython.items():
        body = body.replace(cpp, python)
    body = replace_fors(body)
    body, parallel_add = replace_atomic_adds(body)
    body, parallel_min = replace_atomic_mins(body)
    parallel = parallel_add and parallel_min
    result = f'''
def make(self):
    import numpy as np
    import numba
    @numba.njit(parallel={parallel and JIT_FLAGS['parallel']}, error_model='numpy', fastmath=True)
    def {name}(__python_n__, {str(args).replace("'", "").replace('"', '')[1:-1]}):
        for {iter_var} in numba.prange(__python_n__):
            {body}

    return {name}
'''

    return result

from ...numba.conf import JIT_FLAGS
import re


jit_opts = "error_model='numpy', fastmath=True"

cppython = {
    "int ": "",
    "double ": "",
    "float ": "",
    "auto ": "",
    "bool ": "",
    '[] = {': ' = (',
    " {": ":",
    '}; // array': ')',
    "//": "#",
    "||": "or",
    "&&": "and",
    "! ": "not ",
    "&": "",
    "(int64_t)": "np.int64",  # TODO #324 unit test depicting what fails when this is changed to int16
    "(double)": "np.float64",
    "(float)": "np.float32",
    "return;": "continue",
    "void*": "",
    '::': '_',
    '(*': '',
    ')(, )': '',
    'else if': 'elif',
    'printf': 'print'
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
    cpp = cpp.replace("unsigned int*", "") \
              .replace("double*", "") \
              .replace("float*", "") \
              .replace(" ", "") \
              .replace("()", "") \
              .replace("&", "")
    cpp = re.sub(
        r"atomicMin\((\w+)\[(\w+)],\s*__float_as_uint\(([^)]*)\)\);",
        r"np.minimum(\1[\2:\2+1], np.asarray(\3), \1[\2:\2+1]);", cpp)
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


def extract_struct_defs(cpp: str) -> (str, str):
    stop_string = "};"
    structs = []
    names = []

    start = cpp.find("struct ")
    while start > -1:
        stop = cpp.find(stop_string, start)
        struct = cpp[start:stop + len(stop_string)]
        cpp = cpp.replace(struct, '')
        start = cpp.find("struct ", start + 1)
        structs.append(struct[struct.find(':')+1:-2])
        names.append(re.match(r'(struct )(.*)(:)', struct)[2])

    for i, struct in enumerate(structs):
        structs[i] = struct.replace('static __device__ ', f'\n    @numba.njit(parallel=False, {jit_opts})\n    def {names[i]}_')

    return cpp, '\n'.join(structs)


def to_numba(name, args, iter_var, body):
    body = re.sub(r"static_assert\([^;]*;", '', body)
    body = re.sub(r"static_cast<[^;]*>", '', body)
    body = body.replace("\n", "\n    ")
    for cpp, python in cppython.items():
        body = body.replace(cpp, python)
    body = replace_fors(body)
    body, parallel_add = replace_atomic_adds(body)
    body, parallel_min = replace_atomic_mins(body)
    parallel = parallel_add and parallel_min

    body, structs = extract_struct_defs(body)

    result = f'''
def make(self):
    import numpy as np
    from numpy import floor, ceil, exp, log, power, sqrt
    import numba
    
    @numba.njit(parallel=False, {jit_opts})
    def ldexp(a, b):
        return a * 2**b
    
    {structs}
    @numba.njit(parallel={parallel and JIT_FLAGS['parallel']}, {jit_opts})
    def {name}(__python_n__, {str(args).replace("'", "").replace('"', '')[1:-1]}):
        for {iter_var} in numba.prange(__python_n__):
            {body}

    return {name}
'''.replace('};', ')').replace("}", "")

    return result

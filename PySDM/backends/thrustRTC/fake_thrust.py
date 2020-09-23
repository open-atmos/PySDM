"""
Created at 22.09.2020
"""

import types


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
    "&&": "and"
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

    start = cpp.find("<") or cpp.find(">")
    sign = "-" if cpp[start] == ">" else ""
    range_stop = extract_var(cpp, start, ";")

    start = cpp.find("+=") or cpp.find("-=")
    start += 1

    range_step = extract_var(cpp, start, ")")

    return f"for {var} in range({range_start}, {range_stop}, {sign}{range_step}):"


def replace_fors(cpp):
    start = cpp.find("for")
    while start > -1:
        stop = cpp.find(":", start)
        cpp_for = cpp[start:stop+1]
        python_for = for_to_python(cpp_for)
        cpp = cpp.replace(cpp_for, python_for.replace("for", "__python_token__"))
        print(start)
        start = cpp.find("for", start + len(python_for))
        print(start)
    return cpp.replace("__python_token__", "for")

def to_python(name, args, body):
    for cpp, python in cppython.items():
        body = body.replace(cpp, python)
    body = replace_fors(body)
    result = f'''
def make(self):
    import numba
    from numba import prange
    @numba.njit(parallel=True)
    def {name}(__python_n__, {str(args).replace("'", "").replace('"', '')[1:-1]}):
        for i in prange(__python_n__):
            {body}

    return {name}
'''

    return result


class For:
    def __init__(self, args, _, body):
        d = dict()
        exec(to_python("__internal_python_method__", args, body), d)
        self.make = types.MethodType(d["make"], self)
        self.__internal_python_method__ = self.make()

    def launch_n(self, size, args):
        return self.__internal_python_method__(size, *args)

__subtract_body = For(['output', 'subtrahend'], 'i', '''
            output[i] -= subtrahend[i];
    ''')

import numpy as np
output = np.array([4, 2, 4, 2])
subtrahend = np.array([5, 5, 5, 5])
__subtract_body.launch_n(len(output), [output, subtrahend])
print(output, subtrahend)


class FakeThrustRTC:
    @staticmethod
    def DVInt64(number: int):
        return number

    @staticmethod
    def DVDouble(number: float):
        return number

    @staticmethod
    def DVBool(number: int):
        return number

    For = For

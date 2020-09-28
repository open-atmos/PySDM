"""
Created at 22.09.2020
"""

import types
import numpy as np


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

def to_python(name, args, body):
    body = body.replace("\n", "\n    ")
    for cpp, python in cppython.items():
        body = body.replace(cpp, python)
    body = replace_fors(body)
    result = f'''
def make(self):
    import numpy as np
    import numba
    @numba.njit(parallel=True)
    def {name}(__python_n__, {str(args).replace("'", "").replace('"', '')[1:-1]}):
        for i in numba.prange(__python_n__):
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
        return self.__internal_python_method__(size, *(arg.ndarray for arg in args))


class DVRange:
    def __init__(self, ndarray):
        self.ndarray = ndarray
        self.size = lambda: len(self.ndarray)
        self.range = lambda start, stop: DVRange(self.ndarray[start: stop])

    def __setitem__(self, key, value):
        self.ndarray[key] = value

    def __getitem__(self, item):
        return self.ndarray[item]


class DVVector:
    DVRange = DVRange
    DVVector = None

    def __init__(self, ndarray):
        DVVector.DVVector = DVVector
        self.ndarray = ndarray
        self.size = lambda: len(self.ndarray)
        self.range = lambda start, stop: DVRange(self.ndarray[start: stop])
        self.to_host = lambda: np.copy(self.ndarray)

    def __setitem__(self, key, value):
        if isinstance(value, FakeThrustRTC.Number):
            value = value.ndarray
        self.ndarray[key] = value

    def __getitem__(self, item):
        return self.ndarray[item]


class FakeRandRTC:
    class DVRNG:
        def __init__(self):
            self.ndarray = self

        def state_init(self, seed, seed_i, _, states):
            pass


class RNGState:
    def __init__(self):
        self.ndarray = self

    def rand01(self):
        return np.random.uniform(0, 1)

    def __getitem__(self, item):
        return self


class FakeThrustRTC:
    DVVector = DVVector

    class Number:
        def __init__(self, number):
            self.ndarray = number

    @staticmethod
    def DVInt64(number: int):
        return FakeThrustRTC.Number(number)

    @staticmethod
    def DVDouble(number: float):
        return FakeThrustRTC.Number(number)

    @staticmethod
    def DVBool(number: int):
        return FakeThrustRTC.Number(number)

    For = For

    @staticmethod
    def Sort(dvvector):
        dvvector.ndarray[:] = np.sort(dvvector.ndarray)

    @staticmethod
    def Copy(vector_in, vector_out):
        assert vector_out.ndarray.dtype == vector_in.ndarray.dtype
        vector_out.ndarray[:] = vector_in.ndarray

    @staticmethod
    def Fill(vector, value):
        vector[:] = value

    @staticmethod
    def device_vector(elem_cls, size):
        if elem_cls == 'RNGState':
            return RNGState()
        else:
            dtype = float if elem_cls == 'double' else np.int64
            result = np.empty(size, dtype=dtype)
            return DVVector(result)

    @staticmethod
    def device_vector_from_numpy(ndarray):
        result = np.copy(ndarray)
        return DVVector(result)

    @staticmethod
    def Max_Element(dvvector):
        np.amax(dvvector.ndarray)

    @staticmethod
    def Min_Element(dvvector):
        np.amin(dvvector.ndarray)

    @staticmethod
    def DVPermutation(dvvector, idx):
        _length = np.where(idx.ndarray == idx.size())[0]  # TODO why it works with Thrust!?
        length = _length[0] if len(_length) != 0 else idx.size()
        result = dvvector.ndarray[idx.ndarray[:length]]
        return DVVector(result)

    @staticmethod
    def Sort_By_Key(dvvector, key):
        dvvector.ndarray[:] = dvvector.ndarray[np.argsort(key)]


    @staticmethod
    def Count(dvvector, value):
        return np.count_nonzero((dvvector.ndarray == value))

    @staticmethod
    def Reduce(dvvector, start, operator):
        if operator == "+":
            return start.ndarray + dvvector.ndarray.sum()
        if operator == "-":
            return start.ndarray - dvvector.ndarray.sum()
        if operator == "max":
            return max(start.ndarray, np.amax(dvvector.ndarray))

    @staticmethod
    def Plus():
        return "+"

    @staticmethod
    def Minus():
        return "-"

    @staticmethod
    def Maximum():
        return "max"

    @staticmethod
    def Transform_Binary(vec_in1, vec_in2, vec_out, op):
        if op == "+":
            vec_out.ndarray[:] = vec_in1.ndarray + vec_in2.ndarray
        if op == "-":
            vec_out.ndarray[:] = vec_in1.ndarray - vec_in2.ndarray

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
    "&&": "and",
    "(long)": "",
    "floor": "np.floor",
    "return": "continue"
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
    #import numba
    #from numba import prange
    #@numba.njit(parallel=True)
    def {name}(__python_n__, {str(args).replace("'", "").replace('"', '')[1:-1]}):
        for i in range(__python_n__):
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

# __subtract_body = For(['output', 'subtrahend'], 'i', '''
#         output[i] -= subtrahend[i];
#     ''')
#
import numpy as np
# output = np.array([4, 2, 4, 2])
# subtrahend = np.array([5, 5, 5, 5])
# __subtract_body.launch_n(len(output), [output, subtrahend])
# print(output, subtrahend)


class DVRange:
    def __init__(self, ndarray):
        self.ndarray = ndarray
        self.size = lambda: len(self.ndarray)
        self.range = lambda start, stop: DVRange(self.ndarray[start: stop])

    def __setitem__(self, key, value):
        # key = key if isinstance(key, slice) else int(key)
        self.ndarray[key] = value  # TODO

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
        self.to_host = lambda: self.ndarray

    def __setitem__(self, key, value):
        key = key if isinstance(key, slice) else int(key)
        self.ndarray[key] = value  # TODO

    def __getitem__(self, item):
        item = item if isinstance(item, slice) else int(item)
        return self.ndarray[int(item)]  # TODO

import numba


class FakeRandRTC:
    class DVRNG:
        def __init__(self):
            pass

        def state_init(self, seed, seed_i, _, states):
            pass


class RNGState:
    def __init__(self):
        pass

    def rand01(self):
        return np.random.uniform(0, 1)

class FakeThrustRTC:
    DVVector = DVVector

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

    @staticmethod
    def Sort(dvvector):
        np.sort(dvvector.ndarray)

    @staticmethod
    def Copy(vector_in, vector_out):
        vector_out.ndarray[:] = vector_in.ndarray

    @staticmethod
    def Fill(vector, value):
        vector[:] = value

    @staticmethod
    def device_vector(elem_cls, size):
        if elem_cls == 'RNGState':
            return [RNGState() for _ in range(size)]
        else:
            dtype = float if elem_cls == 'double' else int
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
        return dvvector.ndarray[idx.ndarray]

    @staticmethod
    def Sort_By_Key(dvvector, key):
        dvvector.ndarray[:] = dvvector.ndarray[np.argsort(key)]


    @staticmethod
    def Count(dvvector, value):
        return np.count_nonzero((dvvector.ndarray == value))

    @staticmethod
    def Reduce(dvvector, start, operator):
        if operator == "+":
            return start + dvvector.ndarray.sum()
        if operator == "-":
            return start - dvvector.ndarray.sum()
        if operator == "max":
            return max(start, np.amax(dvvector.ndarray))

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

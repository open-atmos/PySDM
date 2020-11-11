"""
Created at 22.09.2020
"""

import types
import numpy as np
from .cpp2python import to_numba
from numba.core.errors import NumbaError
import warnings


class FakeThrustRTC:
    class DVRange:
        def __init__(self, ndarray):
            self.ndarray = ndarray
            self.size = lambda: len(self.ndarray)
            self.range = lambda start, stop: FakeThrustRTC.DVRange(self.ndarray[start: stop])

        def __setitem__(self, key, value):
            self.ndarray[key] = value

        def __getitem__(self, item):
            return self.ndarray[item]

    class DVVector:
        DVRange = None
        DVVector = None

        def __init__(self, ndarray):
            FakeThrustRTC.DVVector.DVVector = FakeThrustRTC.DVVector
            FakeThrustRTC.DVVector.DVRange = FakeThrustRTC.DVRange
            self.ndarray = ndarray
            self.size = lambda: len(self.ndarray)
            self.range = lambda start, stop: FakeThrustRTC.DVRange(self.ndarray[start: stop])
            self.to_host = lambda: np.copy(self.ndarray)

        def __setitem__(self, key, value):
            if isinstance(value, FakeThrustRTC.Number):
                value = value.ndarray
            self.ndarray[key] = value

        def __getitem__(self, item):
            return self.ndarray[item]

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

    class For:
        def __init__(self, args, _, body):
            d = dict()
            self.code = to_numba("__internal_python_method__", args, body)
            exec(self.code, d)
            self.make = types.MethodType(d["make"], self)
            self.__internal_python_method__ = self.make()

        def launch_n(self, size, args):
            try:
                result = self.__internal_python_method__(size, *(arg.ndarray for arg in args))
            except NumbaError as error:
                warnings.warn(f"NumbaError occurred while JIT-compiling: {self.code}")
                raise error
            return result

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
        dtype = float if elem_cls == 'double' else np.int64
        result = np.empty(size, dtype=dtype)
        return FakeThrustRTC.DVVector(result)

    @staticmethod
    def device_vector_from_numpy(ndarray):
        result = np.copy(ndarray)
        return FakeThrustRTC.DVVector(result)

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
        return FakeThrustRTC.DVVector(result)

    @staticmethod
    def Count(dvvector, value):
        unique, counts = np.unique(dvvector.ndarray, return_counts=True)
        results = dict(zip(unique, counts))
        if value.ndarray in results:
            return results[value.ndarray]
        else:
            return 0

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

    @staticmethod
    def Wait():
        pass

import types
import warnings

import numpy as np
from numba.core.errors import NumbaError

from .cpp2python import to_numba


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
    def DVBool(number: bool):
        return FakeThrustRTC.Number(number)

    @staticmethod
    def DVFloat(number: float):
        return FakeThrustRTC.Number(number)

    class For:
        def __init__(self, args, iter_var, body):
            d = dict()
            self.code = to_numba("__internal_python_method__", args, iter_var, body)
            exec(self.code, d)
            self.make = types.MethodType(d["make"], self)
            self.__internal_python_method__ = self.make()

        def launch_n(self, size, args):
            if size == 0:
                raise SystemError("An internal error happened :) (size==0).")
            try:
                result = self.__internal_python_method__(size, *(arg.ndarray for arg in args))
            except (NumbaError, IndexError) as error:
                warnings.warn(f"NumbaError occurred while JIT-compiling: {self.code}")
                raise error
            return result

    @staticmethod
    def Sort(dvvector):
        dvvector.ndarray[:] = np.sort(dvvector.ndarray)

    @staticmethod
    def Copy(vector_in, vector_out):
        if vector_out.ndarray.dtype != vector_in.ndarray.dtype:
            raise ValueError(f"Incompatible types {vector_out.ndarray.dtype} and {vector_in.ndarray.dtype}")
        vector_out.ndarray[:] = vector_in.ndarray

    @staticmethod
    def Fill(vector, value):
        vector[:] = value

    @staticmethod
    def Find(vector, value):
        for i in range(len(vector.ndarray)):
            if vector[i] == value.ndarray:
                return i
        return None

    @staticmethod
    def device_vector(elem_cls, size):
        if elem_cls == 'double':
            dtype = np.float64
        elif elem_cls == 'float':
            dtype = np.float32
        elif elem_cls == 'int64_t':
            dtype = np.int64
        elif elem_cls == 'uint64_t':
            dtype = np.uint64
        elif elem_cls == 'bool':
            dtype = np.bool_
        else:
            raise NotImplementedError(f'Unsupported type {elem_cls}')
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
        _length = np.where(idx.ndarray == idx.size())[0]
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
        if operator == "min":
            return min(start.ndarray, np.amin(dvvector.ndarray))

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
    def Minimum():
        return "min"

    @staticmethod
    def Transform_Binary(vec_in1, vec_in2, vec_out, op):
        if op == "+":
            vec_out.ndarray[:] = vec_in1.ndarray + vec_in2.ndarray
        if op == "-":
            vec_out.ndarray[:] = vec_in1.ndarray - vec_in2.ndarray

    @staticmethod
    def Wait():
        pass

    @staticmethod
    def Sort_By_Key(keys, values):
        values.ndarray[:] = values.ndarray[np.argsort(keys.ndarray)]
        # TODO #328 Thrust sorts keys as well

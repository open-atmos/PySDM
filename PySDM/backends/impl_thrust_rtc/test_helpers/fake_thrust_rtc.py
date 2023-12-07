"""
Numba-based implementation of ThrustRTC API involving translation of C++ code
 to njitted (and multi-threaded) Python code - a hacky workaround enabling
 testing ThrustRTC code on machines with no GPU/CUDA
"""

# pylint: disable=no-member,unsupported-assignment-operation,unsubscriptable-object,no-value-for-parameter
import types
import warnings

import numpy as np
from numba.core.errors import NumbaError

from .cpp2python import to_numba


class FakeThrustRTC:  # pylint: disable=too-many-public-methods
    class DVRange:
        def __init__(self, ndarray):
            self.ndarray = ndarray
            self.size = lambda: len(self.ndarray)
            self.range = lambda start, stop: FakeThrustRTC.DVRange(
                self.ndarray[start:stop]
            )

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
            self.ndarray: np.ndarray = ndarray
            self.size = lambda: len(self.ndarray)
            self.range = lambda start, stop: FakeThrustRTC.DVRange(
                self.ndarray[start:stop]
            )
            self.to_host = lambda: np.copy(self.ndarray)

        def __setitem__(self, key, value):
            if isinstance(value, FakeThrustRTC.Number):
                value = value.ndarray
            self.ndarray[key] = value

        def __getitem__(self, item):
            return self.ndarray[item]

        def name_elem_cls(self):
            return {"f": "float", "d": "double", "l": "long"}[
                np.ctypeslib.as_ctypes_type(self.ndarray.dtype)._type_
            ]

    class Number:  # pylint: disable=too-few-public-methods
        def __init__(self, number):
            self.ndarray = number

    @staticmethod
    def DVInt64(number: int):  # pylint: disable=invalid-name
        assert isinstance(number, int)
        return FakeThrustRTC.Number(number)

    @staticmethod
    def DVDouble(number: float):  # pylint: disable=invalid-name
        return FakeThrustRTC.Number(number)

    @staticmethod
    def DVBool(number: bool):  # pylint: disable=invalid-name
        assert isinstance(number, bool)
        return FakeThrustRTC.Number(number)

    @staticmethod
    def DVFloat(number: float):  # pylint: disable=invalid-name
        return FakeThrustRTC.Number(number)

    class For:  # pylint: disable=too-few-public-methods
        def __init__(self, param_names, name_iter, body):
            for name in param_names:
                assert "," not in name
            d = {}
            self.code = to_numba(
                "__internal_python_method__", param_names, name_iter, body
            )
            exec(self.code, d)  # pylint: disable=exec-used
            self.make = types.MethodType(d["make"], self)
            self.__internal_python_method__ = (
                self.make()  # pylint: disable=not-callable
            )

        def launch_n(self, n, args):
            if n == 0:
                raise SystemError("An internal error happened :) (size==0).")
            try:
                result = self.__internal_python_method__(
                    n, *(arg.ndarray for arg in args)
                )
            except (NumbaError, IndexError) as error:
                warnings.warn(
                    f"Error occurred while JIT-compiling/executing: {self.code}"
                )
                raise error
            return result

    @staticmethod
    def Sort(dvvector: DVVector):  # pylint: disable=invalid-name
        dvvector.ndarray[:] = np.sort(dvvector.ndarray)

    @staticmethod
    def Copy(vector_in: DVVector, vector_out: DVVector):  # pylint: disable=invalid-name
        if vector_out.ndarray.dtype != vector_in.ndarray.dtype:
            raise ValueError(
                f"Incompatible types {vector_out.ndarray.dtype} and {vector_in.ndarray.dtype}"
            )
        vector_out.ndarray[:] = vector_in.ndarray

    @staticmethod
    def Fill(vector, value):  # pylint: disable=invalid-name
        vector[:] = value

    @staticmethod
    def Find(vector: DVVector, value):  # pylint: disable=invalid-name
        for i in range(len(vector.ndarray)):
            if vector[i] == value.ndarray:
                return i
        return None

    @staticmethod
    def device_vector(elem_cls, size):
        if not size > 0:
            raise ValueError("size must be >0")
        if elem_cls == "double":
            dtype = np.float64
        elif elem_cls == "float":
            dtype = np.float32
        elif elem_cls == "int64_t":
            dtype = np.int64
        elif elem_cls == "uint64_t":
            dtype = np.uint64
        elif elem_cls == "bool":
            dtype = np.bool_
        else:
            raise NotImplementedError(f"Unsupported type {elem_cls}")
        result = np.full(
            size, np.nan if elem_cls in ("float", "double") else -666, dtype=dtype
        )
        return FakeThrustRTC.DVVector(result)

    @staticmethod
    def device_vector_from_numpy(ndarray):
        result = np.copy(ndarray)
        return FakeThrustRTC.DVVector(result)

    @staticmethod
    def Max_Element(dvvector):  # pylint: disable=invalid-name
        np.amax(dvvector.ndarray)

    @staticmethod
    def Min_Element(dvvector):  # pylint: disable=invalid-name
        np.amin(dvvector.ndarray)

    @staticmethod
    def DVPermutation(dvvector, idx):  # pylint: disable=invalid-name
        _length = np.where(idx.ndarray == idx.size())[0]
        length = _length[0] if len(_length) != 0 else idx.size()
        result = dvvector.ndarray[idx.ndarray[:length]]
        return FakeThrustRTC.DVVector(result)

    @staticmethod
    def Count(dvvector, value):  # pylint: disable=invalid-name
        unique, counts = np.unique(dvvector.ndarray, return_counts=True)
        results = dict(zip(unique, counts))
        if value.ndarray in results:
            return results[value.ndarray]
        return 0

    @staticmethod
    def Reduce(dvvector, start, operator):  # pylint: disable=invalid-name
        if operator == "+":
            return start.ndarray + dvvector.ndarray.sum()
        if operator == "-":
            return start.ndarray - dvvector.ndarray.sum()
        if operator == "max":
            return max(start.ndarray, np.amax(dvvector.ndarray))
        if operator == "min":
            return min(start.ndarray, np.amin(dvvector.ndarray))
        raise NotImplementedError()

    @staticmethod
    def Plus():  # pylint: disable=invalid-name
        return "+"

    @staticmethod
    def Minus():  # pylint: disable=invalid-name
        return "-"

    @staticmethod
    def Maximum():  # pylint: disable=invalid-name
        return "max"

    @staticmethod
    def Minimum():  # pylint: disable=invalid-name
        return "min"

    @staticmethod
    def Transform_Binary(vec_in1, vec_in2, vec_out, op):  # pylint: disable=invalid-name
        if op == "+":
            vec_out.ndarray[:] = vec_in1.ndarray + vec_in2.ndarray
        elif op == "-":
            vec_out.ndarray[:] = vec_in1.ndarray - vec_in2.ndarray
        else:
            raise NotImplementedError()

    @staticmethod
    def Wait():  # pylint: disable=invalid-name
        pass

    @staticmethod
    def Sort_By_Key(keys, values):  # pylint: disable=invalid-name
        values.ndarray[:] = values.ndarray[np.argsort(keys.ndarray)]
        # TODO #328 Thrust sorts keys as well

    @staticmethod
    def Set_Kernel_Debug(_):  # pylint: disable=invalid-name
        pass

    @staticmethod
    def Set_Verbose(_):  # pylint: disable=invalid-name
        pass

    @staticmethod
    def Get_PTX_Arch():  # pylint: disable=invalid-name
        return 666

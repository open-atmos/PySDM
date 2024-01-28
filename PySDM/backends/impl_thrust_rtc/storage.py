"""
storage internals for the ThrustRTC backend
"""

import numpy as np

from PySDM.backends.impl_common.storage_utils import (
    StorageBase,
    StorageSignature,
    empty,
    get_data_from_ndarray,
)
from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS, trtc
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust


def make_storage_class(BACKEND):  # pylint: disable=too-many-statements
    class Impl:
        @staticmethod
        def thrust(obj):
            if isinstance(obj, tuple):
                result = tuple(Impl.thrust(o) for o in obj)
            elif hasattr(obj, "data"):
                result = obj.data
            elif isinstance(obj, float):
                result = BACKEND._get_floating_point(obj)
            elif isinstance(obj, int):
                result = trtc.DVInt64(obj)
            else:
                raise ValueError(f"Cannot upload {obj} to device.")
            return result

        __add_elementwise_body = trtc.For(
            ("output", "addend"),
            "i",
            """
                output[i] += addend[i];
            """,
        )

        __add_elementwise_with_multiplier_body = trtc.For(
            ("output", "addend", "multiplier"),
            "i",
            """
                output[i] += multiplier * addend[i];
            """,
        )

        __add_body = trtc.For(
            ["output", "addend"],
            "i",
            """
                output[i] += addend;
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def add(output, addend):
            args = (output, addend)
            if hasattr(addend, "data"):
                loop = Impl.__add_elementwise_body
            elif (
                isinstance(addend, tuple)
                and len(addend) == 3
                and isinstance(addend[0], float)
                and addend[1] == "*"
                and isinstance(addend[2], Storage)
            ):
                loop = Impl.__add_elementwise_with_multiplier_body
                args = (output, addend[2], addend[0])
            else:
                loop = Impl.__add_body
            loop.launch_n(n=output.shape[0], args=Impl.thrust(args))

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def amin(data):
            return trtc.Reduce(data, Impl.thrust(np.inf), trtc.Minimum())

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def amax(data):
            return trtc.Reduce(data, Impl.thrust(-np.inf), trtc.Maximum())

        __row_modulo_body = trtc.For(
            ("output", "divisor", "length"),
            "i",
            """
                auto d = (int64_t)(i / length);
                output[i] %= divisor[d];
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def row_modulo(output, divisor):
            Impl.__row_modulo_body.launch_n(
                output.shape[0], Impl.thrust((output, divisor, output.shape[1]))
            )

        __floor_body = trtc.For(
            ("arr",),
            "i",
            """
                if (arr[i] >= 0) {
                    arr[i] = (int64_t)(arr[i]);
                }
                else {
                    auto old = arr[i];
                    arr[i] = (int64_t)(arr[i]);
                    if (old != arr[i]) {
                        arr[i] -= 1;
                    }
                }
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def floor(output):
            Impl.__floor_body.launch_n(output.shape[0], Impl.thrust((output,)))

        __floor_out_of_place_body = trtc.For(
            ("output", "input_data"),
            "i",
            """
                if (input_data[i] >= 0) {
                    output[i] = (int64_t)(input_data[i]);
                }
                else {
                    output[i] = (int64_t)(input_data[i]);
                    if (input_data[i] != output[i]) {
                        output[i] -= 1;
                    }
                }
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def floor_out_of_place(output, input_data):
            Impl.__floor_out_of_place_body.launch_n(
                output.shape[0], Impl.thrust((output, input_data))
            )

        __multiply_elementwise_body = trtc.For(
            ("output", "multiplier"),
            "i",
            """
                output[i] *= multiplier[i];
            """,
        )

        __multiply_body = trtc.For(
            ["output", "multiplier"],
            "i",
            """
                output[i] *= multiplier;
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def multiply(output, multiplier):
            if hasattr(multiplier, "data"):
                loop = Impl.__multiply_elementwise_body
            else:
                loop = Impl.__multiply_body
            loop.launch_n(output.shape[0], Impl.thrust((output, multiplier)))

        __truediv_elementwise_body = trtc.For(
            ("output", "divisor"),
            "i",
            """
                output[i] /= divisor[i];
            """,
        )

        __truediv_body = trtc.For(
            ["output", "divisor"],
            "i",
            """
                output[i] /= divisor;
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def truediv(output, divisor):
            if hasattr(divisor, "data"):
                loop = Impl.__truediv_elementwise_body
            else:
                loop = Impl.__truediv_body
            loop.launch_n(output.shape[0], Impl.thrust((output, divisor)))

        __multiply_out_of_place_elementwise_body = trtc.For(
            ("output", "multiplicand", "multiplier"),
            "i",
            """
                output[i] = multiplicand[i] * multiplier[i];
            """,
        )

        __multiply_out_of_place_body = trtc.For(
            ("output", "multiplicand", "multiplier"),
            "i",
            """
                output[i] = multiplicand[i] * multiplier;
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def multiply_out_of_place(output, multiplicand, multiplier):
            if hasattr(multiplier, "data"):
                loop = Impl.__multiply_out_of_place_elementwise_body
            elif isinstance(multiplier, float):
                loop = Impl.__multiply_out_of_place_body
            else:
                raise NotImplementedError()
            loop.launch_n(
                output.shape[0], Impl.thrust((output, multiplicand, multiplier))
            )

        __divide_out_of_place_elementwise_body = trtc.For(
            ("output", "dividend", "divisor"),
            "i",
            """
                output[i] = dividend[i] / divisor[i];
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def divide_out_of_place(output, dividend, divisor):
            if hasattr(divisor, "data"):
                loop = Impl.__divide_out_of_place_elementwise_body
            else:
                raise NotImplementedError()
            loop.launch_n(output.shape[0], Impl.thrust((output, dividend, divisor)))

        __sum_out_of_place_elementwise_body = trtc.For(
            ("output", "a", "b"),
            "i",
            """
                output[i] = a[i] + b[i];
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def sum_out_of_place(output, a, b):
            if hasattr(a, "data"):
                loop = Impl.__sum_out_of_place_elementwise_body
            else:
                raise NotImplementedError()
            loop.launch_n(output.shape[0], Impl.thrust((output, a, b)))

        __power_body = trtc.For(
            ("output", "exponent"),
            "i",
            """
                output[i] = pow(output[i], exponent);
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def power(output, exponent: int):
            if exponent == 1:
                return
            Impl.__power_body.launch_n(
                output.shape[0], Impl.thrust((output, float(exponent)))
            )

        __subtract_body = trtc.For(
            ("output", "subtrahend"),
            "i",
            """
                output[i] -= subtrahend[i];
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def subtract(output, subtrahend):
            Impl.__subtract_body.launch_n(
                output.shape[0], Impl.thrust((output, subtrahend))
            )

        __divide_if_not_zero_body = trtc.For(
            param_names=("output", "divisor"),
            name_iter="i",
            body="""
            if (divisor[i] != 0.0) {
                output[i] /= divisor[i];
            }
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def divide_if_not_zero(output, divisor):
            Impl.__divide_if_not_zero_body.launch_n(
                n=(output.shape[0]), args=(Impl.thrust((output, divisor)))
            )

        __exp_body = trtc.For(
            ("output",),
            "i",
            """
                output[i] = exp(output[i]);
            """,
        )

        @staticmethod
        @nice_thrust(**NICE_THRUST_FLAGS)
        def exp(output):
            Impl.__exp_body.launch_n(output.shape[0], Impl.thrust((output,)))

    class Storage(StorageBase):
        FLOAT = BACKEND._get_np_dtype()
        INT = np.int64
        BOOL = np.bool_

        def __getitem__(self, item):
            dim = len(self.shape)
            if isinstance(item, slice):
                start = item.start or 0
                stop = item.stop or self.shape[0]
                if dim == 1:
                    result_data = self.data.range(start, stop)
                    result_shape = (stop - start,)
                elif dim == 2:
                    result_data = self.data.range(
                        self.shape[1] * start, self.shape[1] * stop
                    )
                    result_shape = (stop - start, self.shape[1])
                else:
                    raise NotImplementedError(
                        "Only 2 or less dimensions array is supported."
                    )
                result = Storage(
                    StorageSignature(result_data, result_shape, self.dtype)
                )
            elif (
                dim == 2
                and isinstance(item, tuple)
                and isinstance(item[0], int)
                and isinstance(item[1], slice)
            ):
                assert item[1].start is None or item[1].start == 0
                assert item[1].stop is None or item[1].stop == self.shape[1]
                assert item[1].step is None or item[1].step == 1
                result_data = self.data.range(
                    self.shape[1] * item[0], self.shape[1] * (item[0] + 1)
                )
                result = Storage(
                    StorageSignature(result_data, (*self.shape[1:],), self.dtype)
                )
            else:
                result = self.to_ndarray()[item]
            return result

        def __setitem__(self, key, value):
            if not (
                isinstance(key, slice)
                and key.start is None
                and key.stop is None
                and key.step is None
            ):
                raise NotImplementedError()
            if (
                hasattr(value, "data")
                and hasattr(value, "shape")
                and len(value.shape) != 0
            ):
                if isinstance(value, np.ndarray):
                    vector = trtc.device_vector_from_numpy(value)
                    trtc.Copy(vector, self.data)
                else:
                    trtc.Copy(value.data, self.data)
            else:
                if isinstance(value, int):
                    dvalue = trtc.DVInt64(value)
                elif isinstance(value, float):
                    dvalue = BACKEND._get_floating_point(value)
                else:
                    raise TypeError("Only Storage, int and float are supported.")
                trtc.Fill(self.data, dvalue)
            return self

        def __iadd__(self, other):
            Impl.add(self, other)
            return self

        def __isub__(self, other):
            Impl.subtract(self, other)
            return self

        def __imul__(self, other):
            Impl.multiply(self, other)
            return self

        def __itruediv__(self, other):
            Impl.truediv(self, other)
            return self

        def __imod__(self, other):
            Impl.row_modulo(self, other)
            return self

        def __ipow__(self, other):
            Impl.power(self, other)
            return self

        def __bool__(self):
            if len(self) == 1:
                result = bool(self.data.to_host()[0] != 0)
            else:
                raise NotImplementedError("Logic value of array is ambiguous.")
            return result

        def _to_host(self):
            if isinstance(self.data, trtc.DVVector.DVRange):
                if self.dtype is self.FLOAT:
                    elem_cls = BACKEND._get_c_type()
                elif self.dtype is self.INT:
                    elem_cls = "int64_t"
                elif self.dtype is self.BOOL:
                    elem_cls = "bool"
                else:
                    raise NotImplementedError()

                data = trtc.device_vector(elem_cls, self.data.size())

                trtc.Copy(self.data, data)
            else:
                data = self.data
            return data.to_host()

        def amin(self):
            return Impl.amin(self.data)

        def amax(self):
            return Impl.amax(self.data)

        def all(self):
            assert self.dtype is self.BOOL
            return self.amin()

        def download(self, target, reshape=False):
            shape = target.shape if reshape else self.shape
            target[:] = np.reshape(self._to_host(), shape)

        @staticmethod
        def _get_empty_data(shape, dtype):
            if dtype in (float, Storage.FLOAT):
                elem_cls = BACKEND._get_c_type()
                dtype = Storage.FLOAT
            elif dtype in (int, Storage.INT):
                elem_cls = "int64_t"
                dtype = Storage.INT
            elif dtype in (bool, Storage.BOOL):
                elem_cls = "bool"
                dtype = Storage.BOOL
            else:
                raise NotImplementedError

            size = int(np.prod(shape))
            if size == 0:
                data = None
            else:
                data = trtc.device_vector(elem_cls, size)
            return StorageSignature(data, shape, dtype)

        @staticmethod
        def empty(shape, dtype):
            return empty(shape, dtype, Storage)

        @staticmethod
        def _get_data_from_ndarray(array):
            return get_data_from_ndarray(
                array=array,
                storage_class=Storage,
                copy_fun=lambda array_astype: trtc.device_vector_from_numpy(
                    array_astype.ravel()
                ),
            )

        @staticmethod
        def from_ndarray(array):
            result = Storage(Storage._get_data_from_ndarray(array))
            return result

        def floor(self, other=None):
            if other is None:
                Impl.floor(self.data)
            else:
                Impl.floor_out_of_place(self, other)
            return self

        def product(self, multiplicand, multiplier):
            Impl.multiply_out_of_place(self, multiplicand, multiplier)
            return self

        def ratio(self, dividend, divisor):
            Impl.divide_out_of_place(self, dividend, divisor)
            return self

        def sum(self, arg_a, arg_b):
            Impl.sum_out_of_place(self, arg_a, arg_b)
            return self

        def ravel(self, other):
            if isinstance(other, Storage):
                trtc.Copy(other.data, self.data)
            else:
                self.data = trtc.device_vector_from_numpy(other.ravel())

        def to_ndarray(self):
            result = self._to_host()
            result = np.reshape(result, self.shape)
            return result

        def urand(self, generator):
            generator(self)

        def upload(self, data):
            trtc.Copy(
                trtc.device_vector_from_numpy(data.astype(self.dtype).ravel()),
                self.data,
            )

        def divide_if_not_zero(self, divisor):
            Impl.divide_if_not_zero(self, divisor)
            return self

        def fill(self, other):
            if isinstance(other, Storage):
                trtc.Copy(other.data, self.data)
            else:
                if isinstance(other, int):
                    dvalue = trtc.DVInt64(other)
                elif isinstance(other, float):
                    dvalue = BACKEND._get_floating_point(other)
                else:
                    raise TypeError("Only Storage, int and float are supported.")
                trtc.Fill(self.data, dvalue)
            return self

        def exp(self):
            Impl.exp(self)
            return self

    return Storage

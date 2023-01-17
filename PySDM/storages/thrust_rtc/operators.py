from typing import Any, Callable

import numpy as np

from PySDM.storages.common.storage import Storage
from PySDM.storages.thrust_rtc.conf import NICE_THRUST_FLAGS, trtc
from PySDM.storages.thrust_rtc.nice_thrust import nice_thrust


class ThrustStorageOperators:
    def __init__(self, floating_point: Callable[[float], Any]):
        self.floating_point = floating_point

    def thrust(self, obj: Any):
        if isinstance(obj, tuple):
            result = tuple(self.thrust(o) for o in obj)
        elif Storage.is_storage(obj):
            result = obj.data
        elif isinstance(obj, float):
            result = self.floating_point(obj)
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

    __add_body = trtc.For(
        ["output", "addend"],
        "i",
        """
            output[i] += addend;
        """,
    )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def add(self, output, addend):
        loop = (
            self.__add_elementwise_body
            if Storage.is_storage(addend)
            else self.__add_body
        )
        loop.launch_n(output.shape[0], self.thrust((output, addend)))

    @nice_thrust(**NICE_THRUST_FLAGS)
    def amin(self, data):
        return trtc.Reduce(data, self.thrust(np.inf), trtc.Minimum())

    __row_modulo_body = trtc.For(
        ("output", "divisor", "length"),
        "i",
        """
            auto d = (int64_t)(i / length);
            output[i] %= divisor[d];
        """,
    )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def row_modulo(self, output, divisor):
        self.__row_modulo_body.launch_n(
            output.shape[0], self.thrust((output, divisor, output.shape[1]))
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

    @nice_thrust(**NICE_THRUST_FLAGS)
    def floor(self, output):
        self.__floor_body.launch_n(output.shape[0], self.thrust((output,)))

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

    @nice_thrust(**NICE_THRUST_FLAGS)
    def floor_out_of_place(self, output, input_data):
        self.__floor_out_of_place_body.launch_n(
            output.shape[0], self.thrust((output, input_data))
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

    @nice_thrust(**NICE_THRUST_FLAGS)
    def multiply(self, output, multiplier):
        if Storage.is_storage(multiplier):
            loop = self.__multiply_elementwise_body
        else:
            loop = self.__multiply_body
        loop.launch_n(output.shape[0], self.thrust((output, multiplier)))

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

    @nice_thrust(**NICE_THRUST_FLAGS)
    def truediv(self, output, divisor):
        if Storage.is_storage(divisor):
            loop = self.__truediv_elementwise_body
        else:
            loop = self.__truediv_body
        loop.launch_n(output.shape[0], self.thrust((output, divisor)))

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

    @nice_thrust(**NICE_THRUST_FLAGS)
    def multiply_out_of_place(self, output, multiplicand, multiplier):
        if Storage.is_storage(multiplier):
            loop = self.__multiply_out_of_place_elementwise_body
        elif isinstance(multiplier, float):
            loop = self.__multiply_out_of_place_body
        else:
            raise NotImplementedError()
        loop.launch_n(output.shape[0], self.thrust((output, multiplicand, multiplier)))

    __divide_out_of_place_elementwise_body = trtc.For(
        ("output", "dividend", "divisor"),
        "i",
        """
            output[i] = dividend[i] / divisor[i];
        """,
    )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def divide_out_of_place(self, output, dividend, divisor):
        if Storage.is_storage(divisor):
            loop = self.__divide_out_of_place_elementwise_body
        else:
            raise NotImplementedError()
        loop.launch_n(output.shape[0], self.thrust((output, dividend, divisor)))

    __sum_out_of_place_elementwise_body = trtc.For(
        ("output", "a", "b"),
        "i",
        """
            output[i] = a[i] + b[i];
        """,
    )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def sum_out_of_place(self, output, a, b):
        if Storage.is_storage(a):
            loop = self.__sum_out_of_place_elementwise_body
        else:
            raise NotImplementedError()
        loop.launch_n(output.shape[0], self.thrust((output, a, b)))

    __power_body = trtc.For(
        ("output", "exponent"),
        "i",
        """
            output[i] = pow(output[i], exponent);
        """,
    )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def power(self, output, exponent: int):
        if exponent == 1:
            return
        self.__power_body.launch_n(
            output.shape[0], self.thrust((output, float(exponent)))
        )

    __subtract_body = trtc.For(
        ("output", "subtrahend"),
        "i",
        """
            output[i] -= subtrahend[i];
        """,
    )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def subtract(self, output, subtrahend):
        self.__subtract_body.launch_n(
            output.shape[0], self.thrust((output, subtrahend))
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

    @nice_thrust(**NICE_THRUST_FLAGS)
    def divide_if_not_zero(self, output, divisor):
        self.__divide_if_not_zero_body.launch_n(
            n=(output.shape[0]), args=(self.thrust((output, divisor)))
        )

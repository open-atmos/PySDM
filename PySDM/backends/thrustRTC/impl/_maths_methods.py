"""
Created at 10.12.2019
"""

from ..conf import trtc
from ._storage_methods import StorageMethods
from PySDM.backends.thrustRTC.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS


class MathsMethods:

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def add(output, addend):
        trtc.Transform_Binary(addend, output, output, trtc.Plus())

    __row_modulo_body = trtc.For(['output', 'divisor', 'length'], "i", '''
        int d = i / length;
        output[i] %= divisor[d];
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def row_modulo(output, divisor):
        length = trtc.DVInt64(output.shape[1])
        MathsMethods.__row_modulo_body.launch_n(output.size(), [output, divisor, length])

    __floor_body = trtc.For(['arr'], "i", '''
        if (arr[i] >= 0) {
            arr[i] = (long) arr[i];
        }
        else {
            auto old = arr[i];
            arr[i] = (long) arr[i];
            if (old != arr[i]) {
                arr[i] -= 1;
            }
        }
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def floor(output):
        MathsMethods.__floor_body.launch_n(output.size(), [output])

    __floor_out_of_place_body = trtc.For(['output', 'input_data'], "i", '''
        output[i] = (long) floor(input_data[i]);
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def floor_out_of_place(output, input_data):
        MathsMethods.__floor_out_of_place_body.launch_n(output.size(), [output, input_data])

    __multiply_elementwise_body = trtc.For(['output', 'multiplier'], "i", '''
        output[i] *= multiplier[i];
        ''')

    __multiply_body = trtc.For(['output', 'multiplier'], "i", '''
        output[i] *= multiplier;
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def multiply(output, multiplier):
        if isinstance(multiplier, StorageMethods.storage):
            loop = MathsMethods.__multiply_elementwise_body
            device_multiplier = multiplier
        elif isinstance(multiplier, float):
            loop = MathsMethods.__multiply_body
            device_multiplier = trtc.DVDouble(multiplier)
        elif isinstance(multiplier, int):
            loop = MathsMethods.__multiply_body
            device_multiplier = trtc.DVInt64(multiplier)
        else:
            raise NotImplementedError()
        loop.launch_n(output.size(), [output, device_multiplier])

    __multiply_out_of_place_elementwise_body = trtc.For(['output', 'multiplicand', 'multiplier'], "i", '''
        output[i] = multiplicand[i] * multiplier[i];
        ''')

    __multiply_out_of_place_body = trtc.For(['output', 'multiplicand', 'multiplier'], "i", '''
        output[i] = multiplicand[i] * multiplier;
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def multiply_out_of_place(output, multiplicand, multiplier):
        if isinstance(multiplier, StorageMethods.storage):
            loop = MathsMethods.__multiply_out_of_place_elementwise_body
            device_multiplier = multiplier
        elif isinstance(multiplier, float):
            loop = MathsMethods.__multiply_out_of_place_body
            device_multiplier = trtc.DVDouble(multiplier)
        else:
            raise NotImplementedError()
        loop.launch_n(output.size(), [output, multiplicand, device_multiplier])

    __power_body = trtc.For(['output', 'exponent'], "i", '''
        output[i] = pow(output[i], exponent);
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def power(output, exponent):
        if exponent == 1:
            return
        device_exponent = trtc.DVDouble(exponent)
        MathsMethods.__power_body.launch_n(output.size(), [output, device_exponent])

    __subtract_body = trtc.For(['output', 'subtrahend'], 'i', '''
        output[i] -= subtrahend[i];
        ''')

    @staticmethod
    @nice_thrust(**NICE_THRUST_FLAGS)
    def subtract(output, subtrahend):
        MathsMethods.__subtract_body.launch_n(output.size(), [output, subtrahend])

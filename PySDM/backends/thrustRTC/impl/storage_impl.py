from PySDM.backends.thrustRTC.conf import trtc
from PySDM.backends.thrustRTC.impl.nice_thrust import nice_thrust
from PySDM.backends.thrustRTC.conf import NICE_THRUST_FLAGS
from PySDM.backends.thrustRTC.impl.precision_resolver import PrecisionResolver
import numpy as np


def thrust(obj):
    if isinstance(obj, list):
        result = [thrust(o) for o in obj]
    elif hasattr(obj, 'data'):
        result = obj.data
    elif isinstance(obj, float):
        result = PrecisionResolver.get_floating_point(obj)
    elif isinstance(obj, int):
        result = trtc.DVInt64(obj)
    else:
        raise ValueError(f"Cannot upload {obj} to device.")
    return result


@nice_thrust(**NICE_THRUST_FLAGS)
def add(output, addend):
    trtc.Transform_Binary(thrust(addend), thrust(output), thrust(output), trtc.Plus())


@nice_thrust(**NICE_THRUST_FLAGS)
def amin(data):
    return trtc.Reduce(data, thrust(np.inf), trtc.Minimum())


__row_modulo_body = trtc.For(['output', 'divisor', 'length'], "i", '''
        auto d = (int64_t)(i / length);
        output[i] %= divisor[d];
    ''')


@nice_thrust(**NICE_THRUST_FLAGS)
def row_modulo(output, divisor):
    __row_modulo_body.launch_n(output.shape[0], thrust([output, divisor, output.shape[1]]))


__floor_body = trtc.For(['arr'], "i", '''
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
    ''')


@nice_thrust(**NICE_THRUST_FLAGS)
def floor(output):
    __floor_body.launch_n(output.shape[0], thrust([output]))


__floor_out_of_place_body = trtc.For(['output', 'input_data'], "i", '''
        if (input_data[i] >= 0) {
            output[i] = (int64_t)(input_data[i]);
        }
        else {
            output[i] = (int64_t)(input_data[i]);
            if (input_data[i] != output[i]) {
                output[i] -= 1;
            }
        }
    ''')


@nice_thrust(**NICE_THRUST_FLAGS)
def floor_out_of_place(output, input_data):
    __floor_out_of_place_body.launch_n(output.shape[0], thrust([output, input_data]))


__multiply_elementwise_body = trtc.For(['output', 'multiplier'], "i", '''
        output[i] *= multiplier[i];
    ''')

__multiply_body = trtc.For(['output', 'multiplier'], "i", '''
        output[i] *= multiplier;
    ''')


@nice_thrust(**NICE_THRUST_FLAGS)
def multiply(output, multiplier):
    if hasattr(multiplier, 'data'):
        loop = __multiply_elementwise_body
    else:
        loop = __multiply_body
    loop.launch_n(output.shape[0], thrust([output, multiplier]))


__truediv_elementwise_body = trtc.For(['output', 'multiplier'], "i", '''
        output[i] /= multiplier[i];
    ''')

__truediv_body = trtc.For(['output', 'multiplier'], "i", '''
        output[i] /= multiplier;
    ''')


@nice_thrust(**NICE_THRUST_FLAGS)
def truediv(output, multiplier):
    if hasattr(multiplier, 'data'):
        loop = __truediv_elementwise_body
    else:
        loop = __truediv_body
    loop.launch_n(output.shape[0], thrust([output, multiplier]))


__multiply_out_of_place_elementwise_body = trtc.For(['output', 'multiplicand', 'multiplier'], "i", '''
        output[i] = multiplicand[i] * multiplier[i];
    ''')

__multiply_out_of_place_body = trtc.For(['output', 'multiplicand', 'multiplier'], "i", '''
        output[i] = multiplicand[i] * multiplier;
    ''')


@nice_thrust(**NICE_THRUST_FLAGS)
def multiply_out_of_place(output, multiplicand, multiplier):
    if hasattr(multiplier, 'data'):
        loop = __multiply_out_of_place_elementwise_body
    elif isinstance(multiplier, float):
        loop = __multiply_out_of_place_body
    else:
        raise NotImplementedError()
    loop.launch_n(output.shape[0], thrust([output, multiplicand, multiplier]))


__divide_out_of_place_elementwise_body = trtc.For(['output', 'dividend', 'divisor'], "i", '''
        output[i] = dividend[i] / divisor[i];
    ''')


@nice_thrust(**NICE_THRUST_FLAGS)
def divide_out_of_place(output, dividend, divisor):
    if hasattr(divisor, 'data'):
        loop = __divide_out_of_place_elementwise_body
    else:
        raise NotImplementedError()
    loop.launch_n(output.shape[0], thrust([output, dividend, divisor]))


__power_body = trtc.For(['output', 'exponent'], "i", '''
        output[i] = pow(output[i], exponent);
    ''')


@nice_thrust(**NICE_THRUST_FLAGS)
def power(output, exponent: int):
    if exponent == 1:
        return
    __power_body.launch_n(output.shape[0], thrust([output, float(exponent)]))


__subtract_body = trtc.For(['output', 'subtrahend'], 'i', '''
        output[i] -= subtrahend[i];
    ''')


@nice_thrust(**NICE_THRUST_FLAGS)
def subtract(output, subtrahend):
    __subtract_body.launch_n(output.shape[0], thrust([output, subtrahend]))

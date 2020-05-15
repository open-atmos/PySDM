"""
Created at 10.12.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
import ThrustRTC as trtc
# TODO import CURandRTC as rndrtc
from ._storage_methods import StorageMethods


class MathsMethods:
    chunks = 32

    @staticmethod
    def add(output, addend):
        trtc.Transform_Binary(addend, output, output, trtc.Plus())

    __column_modulo_body = trtc.For(['output', 'divisor', 'col_num'], "i", '''
        int d = i % col_num;
        output[i] %= divisor[d];
        ''')

    @staticmethod
    def column_modulo(output, divisor):
        col_num = trtc.DVInt64(divisor.size())
        MathsMethods.__column_modulo_body.launch_n(output.size(), [output, divisor, col_num])

    __floor_body = trtc.For(['arr'], "i", '''
        if (arr[i] >= 0) 
            arr[i] = (long) arr[i];
        else
        {
            auto old = arr[i];
            arr[i] = (long) arr[i];
            if (old != arr[i])
                arr[i] -= 1;
        }
        ''')

    @staticmethod
    def floor(output):
        MathsMethods.__floor_body.launch_n(output.size(), [output])

    __floor_out_of_place_body = trtc.For(['output', 'input_data'], "i", '''
        if (input_data[i] >= 0) 
            output[i] = (long) input_data[i];
        else
        {
            output[i] = (long) input_data[i];
            if (input_data[i] != output[i])
                output[i] -= 1;
        }
        ''')

    @staticmethod
    def floor_out_of_place(output, input_data):
        MathsMethods.__floor_out_of_place_body.launch_n(output.size(), [output, input_data])

    __multiply_elementwise_body = trtc.For(['output', 'multiplier'], "i", '''
        output[i] *= multiplier[i];
        ''')

    __multiply_body = trtc.For(['output', 'multiplier'], "i", '''
        output[i] *= multiplier;
        ''')

    @staticmethod
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
    def power(output, exponent):
        if exponent == 1:
            return
        device_multiplier = trtc.DVDouble(exponent)
        MathsMethods.__power_body.launch_n(output.size(), [output, device_multiplier])

    @staticmethod
    def subtract(output, subtrahend):
        trtc.Transform_Binary(subtrahend, output, output, trtc.Minus())

    @staticmethod
    def urand(data, seed=None):
        # TODO: print("Numpy import!: ThrustRTC.urand(...)")

        np.random.seed(seed)
        output = np.random.uniform(0, 1, data.shape)
        StorageMethods.upload(output, data)

        # if seed is None:
        #     seed = np.random.randint(2**16)
        # rng = rndrtc.DVRNG()
        #
        # chunks = MathsMethods.chunks
        # ker = trtc.For(['rng', 'vec_rnd'], 'idx', f'''
        #     RNGState state;
        #     rng.state_init({seed}, idx, 0, state);  // initialize a state using the rng object
        #     for (int i=0; i<{chunks}; i++)
        #        vec_rnd[i+idx*{chunks}]=(float)state.rand01();  // generate random number using the rng object
        #     ''')
        #
        # ker.launch_n(data.size() // chunks, [rng, data])
        #
        # if data.size() % MathsMethods.chunks != 0:
        #     start = data.size() - (data.size() % MathsMethods.chunks)
        #     stop = data.size()
        #     data_tail = data.range(start, stop)
        #     ker = trtc.For(['rng', 'vec_rnd'], 'idx', f'''
        #         RNGState state;
        #         rng.state_init({seed}, {start}+idx, 0, state);  // initialize a state using the rng object
        #         vec_rnd[idx]=(float)state.rand01();  // generate random number using the rng object
        #         ''')
        #
        #     ker.launch_n(stop - start, [rng, data_tail])

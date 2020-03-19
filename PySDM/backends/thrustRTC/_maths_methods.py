"""
Created at 10.12.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import ThrustRTC as trtc
import CURandRTC as rndrtc
from ._storage_methods import StorageMethods


class MathsMethods:

    @staticmethod
    def add(output, addend):
        trtc.Transform_Binary(addend, output, output, trtc.Plus())

    @staticmethod
    def column_modulo(output, divisor):
        loop = trtc.For(['arr', 'divisor'], "i", f'''
                            for (int d=0; d<{divisor.size()}; d++)
                                arr[d + i] = arr[d + i] % divisor[d];
                        ''')
        loop.launch_n(output.shape[0], [output, divisor])

    @staticmethod
    def floor_out_of_place(output, input_data):
        loop = trtc.For(['out', 'in'], "i", '''
                            if (in[i] >= 0) 
                                out[i] = (long) in[i];
                            else
                            {
                                out[i] = (long) in[i];
                                if (in != out[i])
                                    out[i] -= 1;
                            }
                        ''')
        loop.launch_n(output.size(), [output, input_data])

    @staticmethod
    def floor(output):
        loop = trtc.For(['arr'], "i", '''
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
        loop.launch_n(output.size(), [output])

    @staticmethod
    def multiply_out_of_place(output, multiplicand, multiplier):
        if isinstance(multiplier, StorageMethods.storage):
            loop = trtc.For(['output', 'multiplicand', 'multiplier'], "i",
                            "output[i] = multiplicand[i] * multiplier[i];")
            device_multiplier = multiplier
        elif isinstance(multiplier, float):
            loop = trtc.For(['output', 'multiplicand', 'multiplier'], "i",
                            "output[i] = multiplicand[i] * multiplier;")
            device_multiplier = trtc.DVDouble(multiplier)
        else:
            raise NotImplementedError()
        loop.launch_n(output.size(), [output, multiplicand, device_multiplier])

    @staticmethod
    def multiply(output, multiplier):
        if isinstance(multiplier, StorageMethods.storage):
            loop = trtc.For(['output', 'multiplier'], "i", "output[i] *= multiplier[i];")
            device_multiplier = multiplier
        elif isinstance(multiplier, float):
            loop = trtc.For(['output', 'multiplier'], "i", "output[i] *= multiplier;")
            device_multiplier = trtc.DVDouble(multiplier)
        else:
            raise NotImplementedError()
        loop.launch_n(output.size(), [output, device_multiplier])

    @staticmethod
    def power(output, exponent):
        raise NotImplementedError()

    @staticmethod
    def subtract(output, subtrahend):
        trtc.Transform_Binary(subtrahend, output, output, trtc.Minus)

    @staticmethod
    def urand(data):
        rng = rndrtc.DVRNG()
        # TODO: threads vs. blocks
        # TODO: proper state_init
        # TODO: generator choice
        chunks = min(32, data.size())  # TODO!!!
        ker = trtc.For(['rng', 'vec_rnd'], 'idx',
                       f'''
                       RNGState state;
                       rng.state_init(1234, idx, 0, state);  // initialize a state using the rng object
                       for (int i=0; i<{chunks}; i++)
                           vec_rnd[i+idx*{chunks}]=(float)state.rand01(); // generate random number using the rng object
                       ''')

        ker.launch_n(data.size() // chunks, [rng, data])

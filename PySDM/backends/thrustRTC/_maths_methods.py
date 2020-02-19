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
    def add(data_out, data_in):
        trtc.Transform_Binary(data_in, data_out, data_out, trtc.Plus())

    @staticmethod
    def amax(row, idx, length):
        perm_in = trtc.DVPermutation(row, idx)
        index = trtc.Max_Element(perm_in.range(0, length))
        row_idx = idx.get(index)
        result = row.get(row_idx)
        return result

    @staticmethod
    def amin(row, idx, length):
        perm_in = trtc.DVPermutation(row, idx)
        index = trtc.Min_Element(perm_in.range(0, length))
        row_idx = idx.get(index)
        result = row.get(row_idx)
        return result

    @staticmethod
    def column_modulo(data, divisor):
        loop = trtc.For(['arr', 'divisor'], "i", f'''
                            for (int d=0; d<{divisor.size()}; d++)
                                arr[d + i] = arr[d + i] % divisor[d];
                        ''')
        loop.launch_n(data.shape[0], [data, divisor])

    @staticmethod
    def floor(data_out, data_in):
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
        loop.launch_n(data_out.size(), [data_out, data_in])

    @staticmethod
    def floor_in_place(data):
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
        loop.launch_n(data.size(), [data])

    @staticmethod
    def multiply(data, multiplier):
        if isinstance(multiplier, StorageMethods.storage):
            loop = trtc.For(['arr', 'mult'], "i", "arr[i] *= mult[i];")
            mult = multiplier
        elif isinstance(multiplier, float):
            loop = trtc.For(['arr', 'mult'], "i", "arr[i] *= mult;")
            mult = trtc.DVDouble(multiplier)
        else:
            raise NotImplementedError()
        loop.launch_n(data.size(), [data, mult])

    @staticmethod
    def subtract(data_out, data_in):
        trtc.Transform_Binary(data_in, data_out, data_out, trtc.Minus)

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



from ._parameterized import Parameterized


class ConstEb(Parameterized):

    def __init__(self, Eb=1.0):
        super().__init__((Eb, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
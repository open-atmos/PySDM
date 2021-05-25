from ._parameterized import Parameterized


class Electric(Parameterized):
    """
    E.X. Berry 1967
    Cloud Droplet Growth by Collection

    Kernel for electric filed 3000V/cm
    """

    def __init__(self):
        super().__init__((1, 1, -7, 1.78, -20.5, 1.73, .26, 1.47, 1, .82, -0.003, 4.4, 8))

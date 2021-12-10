import inspect
from PySDM.backends import GPU, CPU


class TestCtorDefaults:
    @staticmethod
    def test_gpu_ctor_defaults():
        signature = inspect.signature(GPU.__init__)
        assert signature.parameters['verbose'].default == False
        assert signature.parameters['debug'].default == False
        assert signature.parameters['double_precision'].default == False
        assert signature.parameters['formulae'].default == None

    @staticmethod
    def test_cpu_ctor_defaults():
        signature = inspect.signature(CPU.__init__)
        assert signature.parameters['formulae'].default == None

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest

from PySDM.backends import CPU, GPU


class GPUFP32(GPU):  # pylint: disable=too-many-ancestors
    def __init__(self, formulae=None, **kwargs):
        if "double_precision" in kwargs:
            if kwargs["double_precision"]:
                pytest.skip()
            else:
                del kwargs["double_precision"]
        super().__init__(formulae=formulae, double_precision=False, **kwargs)


class GPUFP64(GPU):  # pylint: disable=too-many-ancestors
    def __init__(self, formulae=None, **kwargs):
        if "double_precision" in kwargs:
            if kwargs["double_precision"]:
                del kwargs["double_precision"]
            else:
                pytest.skip()
        super().__init__(formulae=formulae, double_precision=True, **kwargs)


@pytest.fixture(params=(CPU, GPUFP32, GPUFP64))  # TODO #1144 CPU float
def backend_class(request):
    return request.param

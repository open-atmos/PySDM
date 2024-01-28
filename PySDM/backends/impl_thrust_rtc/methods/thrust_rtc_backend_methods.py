"""
common parent class for all ThrustRTC backend methods classes
"""

from typing import Callable, Optional

from ...impl_common.backend_methods import BackendMethods


class ThrustRTCBackendMethods(BackendMethods):  # pylint: disable=too-few-public-methods
    def __init__(self):
        super().__init__()
        if not hasattr(self, "_conv_function"):
            self._conv_function: Optional[Callable] = None
        if not hasattr(self, "_real_type"):
            self._real_type = None
        if not hasattr(self, "_np_dtype"):
            self._np_dtype = None

    def _get_floating_point(self, number):
        return self._conv_function(number)  # pylint: disable=not-callable

    def _get_c_type(self):
        return self._real_type

    def _get_np_dtype(self):
        return self._np_dtype

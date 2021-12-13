"""
logic around the `PySDM.backend.impl_common.backend_methods.Backend_Methods` - the parent
 class for all backend methods classes
"""


# pylint: disable=too-few-public-methods
class BackendMethods:
    def __init__(self):
        if not hasattr(self, 'formulae'):
            self.formulae = None
        if not hasattr(self, 'Storage'):
            self.Storage = None

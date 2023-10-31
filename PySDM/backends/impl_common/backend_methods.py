"""
logic around the `PySDM.backends.impl_common.backend_methods.BackendMethods` - the parent
 class for all backend methods classes
"""


# pylint: disable=too-few-public-methods
class BackendMethods:
    def __init__(self):
        if not hasattr(self, "formulae"):
            self.formulae = None
        if not hasattr(self, "Storage"):
            self.Storage = None

    def __setattr__(self, attr, value):
        if hasattr(self, "initialised"):
            raise AssertionError()
        super().__setattr__(attr, value)

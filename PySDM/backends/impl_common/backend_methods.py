"""
logic around the `PySDM.backends.impl_common.backend_methods.BackendMethods` - the parent
 class for all backend methods classes
"""


# pylint: disable=too-few-public-methods
class BackendMethods:
    def __init__(self):
        if not hasattr(self, "formulae"):
            self.formulae = None
        if not hasattr(self, "formulae_flattened"):
            self.formulae_flattened = None
        if not hasattr(self, "Storage"):
            self.Storage = None
        if not hasattr(self, "default_jit_flags"):
            self.default_jit_flags = {}

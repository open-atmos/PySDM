"""
logic around the `PySDM.backends.impl_common.backend_methods.BackendMethods` - the parent
 class for all backend methods classes
"""

from pystrict import strict


# pylint: disable=too-few-public-methods
@strict
class BackendMethods:
    def __init__(self):
        if not hasattr(self, "formulae"):
            self.formulae = None
        if not hasattr(self, "Storage"):
            self.Storage = None

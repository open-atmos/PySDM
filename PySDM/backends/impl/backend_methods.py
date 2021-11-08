# pylint: disable=too-few-public-methods
class BackendMethods:
    def __init__(self):
        if not hasattr(self, 'formulae'):
            self.formulae = None

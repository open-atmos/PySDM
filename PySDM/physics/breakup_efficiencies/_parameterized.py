from PySDM.physics import constants as const


class Parameterized():

    def __init__(self, params):
        self.core = None
        self.params = params
        
    def register(self, builder):
        self.core = builder.core
        builder.request_attribute('radius')

    def __call__(self, output, is_first_in_pair):
        self.core.backend.linear_collection_efficiency(
            self.params, output, self.core.particles['radius'], is_first_in_pair, const.si.um)
        output **= 2


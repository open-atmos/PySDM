"""
Created at 05.21.2021
"""

class ConstEc():

    def __init__(self, Ec=1.0):
        self.Ec = Ec
        self.core = None

    def register(self, builder):
        self.core = builder.core
    
    def __call__(self, output, is_first_in_pair):
        output.data[:] = self.Ec
        print(output.data)
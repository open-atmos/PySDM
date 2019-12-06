"""
Created at 21.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


class DummyEnvironment:
    def __init__(self, _, courant_field_data):
        self.courant_field_data = courant_field_data

    def get_courant_field_data(self):
        return self.courant_field_data

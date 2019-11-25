class DummyEnvironment:
    def __init__(self, _, grid, courant_field_data):
        self.grid = grid
        self.courant_field_data = courant_field_data

    def get_courant_field_data(self):
        return self.courant_field_data
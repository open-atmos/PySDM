class Table:
    def __getitem__(self, item):
        return self._data[item]

    def items(self):
        return self._data.items()

    def __init__(self, volume, data):
        self._data = data
        self.volume = volume

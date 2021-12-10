class Attribute:

    def __init__(self, builder, name, dtype=float, size=0):
        self.particulator = builder.particulator
        self.timestamp: int = 0
        self.data = None
        self.dtype = dtype
        self.size = size
        self.name = name
        self.formulae = self.particulator.formulae

    def allocate(self, idx):
        if self.size >= 1:
            self.data = self.particulator.IndexedStorage.empty(idx, (self.size, self.particulator.n_sd), dtype=self.dtype)
        else:
            self.data = self.particulator.IndexedStorage.empty(idx, (self.particulator.n_sd,), dtype=self.dtype)

    def set_data(self, data):
        self.data = data

    def get(self):
        self.update()
        return self.data

    def update(self):
        pass

    def mark_updated(self):
        self.timestamp += 1

    def __str__(self):
        return self.name

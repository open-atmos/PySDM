import numpy as np


class BasicSimulation:
    def __init__(self, particulator):
        self.particulator = particulator

    def _save(self, output):
        for k, v in self.particulator.products.items():
            value = v.get()
            if isinstance(value, np.ndarray) and value.shape[0] == 1:
                value = value[0]
            output[k].append(value)

    def _run(self, nt, steps_per_output_interval):
        output = {k: [] for k in self.particulator.products}
        self._save(output)
        for _ in range(0, nt + 1, steps_per_output_interval):
            self.particulator.run(steps=steps_per_output_interval)
            self._save(output)
        return output

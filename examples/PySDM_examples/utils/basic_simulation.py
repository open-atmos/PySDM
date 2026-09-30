import numpy as np


class BasicSimulation:
    def __init__(self, particulator, output_attributes=None):
        for k in output_attributes or []:
            assert k in particulator.attributes
        self.particulator = particulator
        self.output_attributes = output_attributes

    def _save(self, output):
        for k, v in self.particulator.products.items():
            value = v.get()
            if isinstance(value, np.ndarray) and value.shape[0] == 1:
                value = value[0]
            output[k].append(value)
        for k in self.output_attributes or []:
            value = self.particulator.attributes[k].to_ndarray()
            output[k].append(value)

    def _run(self, nt, steps_per_output_interval):
        output = {k: [] for k in self.particulator.products}
        output |= {k: [] for k in self.output_attributes or []}
        self._save(output)
        for _ in range(0, nt + 1, steps_per_output_interval):
            self.particulator.run(steps=steps_per_output_interval)
            self._save(output)
        return {k: np.asarray(v) for k, v in output.items()}

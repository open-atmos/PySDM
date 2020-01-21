class Solver:
    atol = 1e-3
    rtol = 1e-3

    def __init__(self, backend, mean_n_sd_in_cell):
        length = 2 * mean_n_sd_in_cell + 2
        self.y = backend.array(length, dtype=float)  # TODO: list


import numpy as np

class QvThd:
    # TODO: pint, mendeleev,
    R_d = 287.00260752682607  # meter ** 2 / kelvin / second ** 2
    R_v = 461.52982514571187  # meter ** 2 / kelvin / second ** 2
    cp_d = 1005.0  # meter ** 2 / kelvin / second ** 2
    p0 = 1000 * 100
    ARM_C1 = 6.1094 * 100  # Pascal
    ARM_C2 = 17.625
    ARM_C3 = 243.04  # Kelvin
    T0 = 273.15
    eps = 0.6218506191581769

    kappa = R_d / cp_d

    def __init__(self, grid, backend, thd_lambda, qv_lambda, p0, Z):
        self.backend = backend
        self.thd_lambda = thd_lambda
        self.qv_lambda = qv_lambda

        self.n_cell = int(np.prod(grid))
        self.qv = backend.array((self.n_cell,), float)
        self.thd = backend.array((self.n_cell,), float)

        self.RH = backend.array((self.n_cell,), float)
        self.p = backend.array((self.n_cell,), float)
        self.T = backend.array((self.n_cell,), float)

        self.grid = grid

        # TODO (function of z)
        # for z in range(self.grid[1]):
            # self.rhod = backend.from_ndarray()
            # self.rhod0 =  self.p0 ** (self.kappa) / (self.R_d * self.thd)
            # self.rhod[z] = self.rhod0 * np.power(self.kappa * self.rhod0 * self.g * (H - z) + self.p1000 ** (self.kappa) ,
            #                                      (self.kappa - 1))
    def sync(self):
        self.backend.upload(self.qv_lambda().ravel(), self.qv)
        self.backend.upload(self.thd_lambda().ravel(), self.thd)

        # TODO: move to backend
        for i in range(self.n_cell):
            pd = np.power((self.rhod * self.R_d * self.thd[i]) / self.p0 ** self.kappa, 1 / (1 - self.kappa))
            self.T[i] = self.thd[i] * (pd / self.p0) ** (self.kappa)
            R = self.R_v / (1/self.qv[i] + 1) + self.R_d / (1 + self.qv[i])
            self.p[i] = self.rhod * (1 + self.qv[i]) * R * self.T[i]

            # August-Roche-Magnus formula
            pvs = self. ARM_C1 * np.exp((self.ARM_C2 * (self.T[i] - self.T0)) / (self.T[i] - self.T0 + self.ARM_C3))
            self.RH[i] = (self.p[i] - pd) / pvs

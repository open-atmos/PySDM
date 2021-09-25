import numpy as np
from pyevtk.hl import pointsToVTK
from pyevtk.vtk import VtkGroup
import numbers, os


"""
Example of use:

exporter = VTKExporter()

for step in range(settings.n_steps):
    simulation.particulator.run(1)

    exporter.export_particles(simulation.particulator)

"""

class VTKExporter:

    def __init__(self, path='.', particles_filename="sd_points", file_num_len=4, verbose=False):
        self.path = os.path.join(path, 'output')
        
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.particles_file_path = os.path.join(self.path, particles_filename)
        self.num_len = file_num_len
        self.exported_times = {}
        self.verbose = verbose

    def write_pvd(self):
        pvd = VtkGroup(self.particles_file_path)
        for k, v in self.exported_times.items():
            pvd.addFile(k + '.vtu', sim_time=v)
        pvd.save()

    def export_particles(self, particulator):
        path = self.particles_file_path + '_num' + self.add_leading_zeros(particulator.n_steps)
        self.exported_times[path] = particulator.n_steps * particulator.dt
        if self.verbose:
            print("Exporting Particles to vtk, path: " + path)
        payload = {}

        for k in particulator.attributes.keys():
            if len(particulator.attributes[k].shape) != 1:
                tmp = particulator.attributes[k].to_ndarray(raw=True)
                tmp_dict = {k + '[' + str(i) + ']' : tmp[i] for i in range(len(particulator.attributes[k].shape))}

                payload.update(tmp_dict)
            else:
                payload[k] = particulator.attributes[k].to_ndarray(raw=True)

        payload.update({k: np.array(v) for k, v in payload.items() if not (v.flags['C_CONTIGUOUS'] or v.flags['F_CONTIGUOUS'])})

        if particulator.mesh.dimension == 2:
            y = particulator.mesh.size[0]/particulator.mesh.grid[0] * (payload['cell origin[0]'] + payload['position in cell[0]'])
            x = particulator.mesh.size[1]/particulator.mesh.grid[1] * (payload['cell origin[1]'] + payload['position in cell[1]'])
            z = np.full_like(x, 0)
        else:
            raise NotImplementedError("Only 2 dimensions array is supported at the moment.")

        pointsToVTK(path, x, y, z, data = payload)

    def add_leading_zeros(self, a):
        return ''.join(['0' for i in range(self.num_len - len(str(a)))]) + str(a)

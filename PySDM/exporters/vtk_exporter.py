import numpy as np
from pyevtk.hl import pointsToVTK
from pyevtk.vtk import VtkGroup
import numbers, os


"""
Example of use:

exporter = VTKExporter()

for step in range(settings.n_steps):
    simulation.core.run(1)

    exporter.export_particles(simulation.core)

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

    def export_particles(self, core):
        path = self.particles_file_path + '_num' + self.add_leading_zeros(core.n_steps)
        self.exported_times[path] = core.n_steps * core.dt
        if self.verbose:
            print("Exporting Particles to vtk, path: " + path)
        payload = {}

        particles = core.particles
        for k in particles.attributes.keys():
            if len(particles[k].shape) != 1:
                tmp = particles[k].to_ndarray(raw=True)
                tmp_dict = {k + '[' + str(i) + ']' : tmp[i] for i in range(len(particles[k].shape))}

                payload.update(tmp_dict)
            else:
                payload[k] = particles[k].to_ndarray(raw=True)

        payload.update({k: np.array(v) for k, v in payload.items() if not (v.flags['C_CONTIGUOUS'] or v.flags['F_CONTIGUOUS'])})

        if core.mesh.dimension == 2:
            y = core.mesh.size[0]/core.mesh.grid[0] * (payload['cell origin[0]'] + payload['position in cell[0]'])
            x = core.mesh.size[1]/core.mesh.grid[1] * (payload['cell origin[1]'] + payload['position in cell[1]'])
            z = np.full_like(x, 0)
        else:
            raise NotImplementedError("Only 2 dimensions array is supported at the moment.")

        pointsToVTK(path, x, y, z, data = payload)

    def add_leading_zeros(self, a):
        return ''.join(['0' for i in range(self.num_len - len(str(a)))]) + str(a)
    
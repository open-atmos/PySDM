import platform

import numpy as np

if platform.architecture()[0] != "32bit":
    import vtk

    # pylint: disable = import-error, no-name-in-module
    from vtk.util import numpy_support as VN

    # pylint: enable = import-error, no-name-in-module


def readVTK_1d(file):
    if platform.architecture()[0] == "32bit":
        raise NotImplementedError("Not implemented for system arcitucture 32bit!")

    reader = vtk.vtkXMLUnstructuredGridReader()  # pylint: disable = no-member
    reader.SetFileName(file)
    reader.Update()

    vtk_output = reader.GetOutput()

    z = np.zeros(vtk_output.GetNumberOfPoints())
    for i in range(vtk_output.GetNumberOfPoints()):
        z[i] = vtk_output.GetPoint(i)[2]

    data = {}
    data["z"] = z
    for i in range(vtk_output.GetPointData().GetNumberOfArrays()):
        data[vtk_output.GetPointData().GetArrayName(i)] = VN.vtk_to_numpy(
            vtk_output.GetPointData().GetArray(i)
        )

    return data

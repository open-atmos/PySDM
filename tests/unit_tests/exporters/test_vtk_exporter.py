"""checks for VTK exporter"""

from collections import namedtuple

import numpy as np

from PySDM.exporters import VTKExporter


def test_vtk_exporter_copies_product_data(tmp_path):
    """note: since VTK files contain unencoded binary data, we cannot use XML parsers;
    not to introduce a new dependency to PySDM, we read the binary data with NumPy"""
    # arrange
    productc_filename = tmp_path / "prod"
    sut = VTKExporter(products_filename=productc_filename)

    grid = (1, 1)
    arr = np.zeros(shape=grid, dtype=float)

    incr = 666

    def plusplus(arr):
        arr += incr
        return arr

    prod = namedtuple(typename="MockProductA", field_names=("get",))(
        get=lambda: plusplus(arr)
    )

    particulator = namedtuple(
        typename="MockParticulator", field_names=("products", "n_steps", "dt", "mesh")
    )(
        n_steps=1,
        products={
            "a": prod,
            "b": prod,
        },
        dt=0,
        mesh=namedtuple(typename="MockMesh", field_names=("dimension", "grid", "size"))(
            dimension=2,
            grid=grid,
            size=(1, 1),
        ),
    )

    # act
    sut.export_products(particulator)

    # assert
    offsets = (113, 129)
    with open(str(productc_filename) + "_num0000000001.vts", mode="rb") as vtk:
        binary_data = vtk.readlines()[14]
        for i, off in enumerate(offsets):
            assert (
                np.frombuffer(binary_data[off : off + 8], dtype=np.float64)
                == (i + 1) * incr
            )

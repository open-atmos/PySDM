import os
import sys

from open_atmos_jupyter_utils import TemporaryFile
from PySDM_examples.Szumowski_et_al_1998.gui_controller import GUIController
from PySDM_examples.Szumowski_et_al_1998.gui_viewer import GUIViewer
from PySDM_examples.utils.widgets import HTML, Tab, VBox, display

from PySDM.exporters import NetCDFExporter


def launch(settings, simulation, storage):
    ncdf_file = TemporaryFile(".nc")
    ncdf_exporter = NetCDFExporter(
        storage, settings, simulation, ncdf_file.absolute_path
    )

    vtk_file = TemporaryFile(".zip")

    viewer = GUIViewer(storage, settings)
    controller = GUIController(simulation, viewer, ncdf_exporter, ncdf_file, vtk_file)

    controller_box = controller.box()

    tabs = Tab([VBox([controller_box, viewer.box()]), settings.box()])
    tabs.set_title(1, "Settings")
    tabs.set_title(0, "Simulation")
    tabs.observe(controller.reinit, "selected_index")

    # https://github.com/googlecolab/colabtools/issues/1302
    hack = (
        "<link"
        ' rel="stylesheet"'
        ' href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"'
        "> "
    )
    if "google.colab" in sys.modules:
        display(HTML(hack))

    display(tabs)

    if "CI" in os.environ:
        controller_box.children[1].click()

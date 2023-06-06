import shutil
from tempfile import TemporaryDirectory
from threading import Thread
from time import sleep

from PySDM_examples.utils.widgets import Button, Checkbox, FloatProgress, HBox

from PySDM.exporters import VTKExporter


class GUIController:
    def __init__(self, simulator, viewer, ncdf_exporter, ncdf_file, vtk_file):
        self.progress = FloatProgress(value=0.0, min=0.0, max=1.0)
        self.button = Button()
        self.link = HBox()
        self.checkbox = Checkbox(
            description="VTK output", indent=False, layout={"width": "125px"}
        )
        self.panic = False
        self.thread = None
        self.simulator = simulator
        self.ncdf_exporter = ncdf_exporter
        self.ncdf_file = ncdf_file
        self.vtk_file = vtk_file
        self.tempdir = None
        self.vtk_exporter = None
        self.viewer = viewer
        self._setup_play()

    def __enter__(self):
        self.panic = False
        self.set_percent(0)

    def __exit__(self, *_):
        if self.panic:
            self._setup_play()
        else:
            self._setup_save()
        self.panic = False
        self.progress.description = " "

    def reinit(self, _=None):
        self.panic = True
        self._setup_play()
        self.progress.value = 0
        self.progress.description = " "
        self.viewer.clear()
        self.link.children = ()
        self.vtk_exporter = None

    def box(self):
        netcdf_box = Checkbox(
            description="netCDF output",
            disabled=True,
            value=True,
            indent=False,
            layout={"width": "125px"},
        )
        return HBox([self.progress, self.button, netcdf_box, self.checkbox, self.link])

    def set_percent(self, value):
        self.progress.value = value

    def _setup_play(self):
        self.button.on_click(self._handle_stop, remove=True)
        self.button.on_click(self._handle_save, remove=True)
        self.button.on_click(self._handle_play)
        self.button.icon = "play"
        self.button.description = "start simulation"
        self.checkbox.disabled = False

    def _setup_stop(self):
        self.button.on_click(self._handle_play, remove=True)
        self.button.on_click(self._handle_save, remove=True)
        self.button.on_click(self._handle_stop)
        self.button.icon = "stop"
        self.button.description = "interrupt"

    def _setup_save(self):
        self.button.icon = "download"
        self.button.description = "save output"
        self.button.on_click(self._handle_stop, remove=True)
        self.button.on_click(self._handle_save)

    def _handle_stop(self, _):
        self.panic = True
        while self.panic:
            sleep(0.1)
        self._setup_play()

    def _handle_play(self, _):
        self._setup_stop()

        self.link.children = ()
        self.progress.value = 0
        self.progress.description = "initialisation"
        self.checkbox.disabled = True

        if self.checkbox.value:
            self.tempdir = TemporaryDirectory()
            self.vtk_exporter = VTKExporter(verbose=False, path=self.tempdir.name)

        self.thread = Thread(target=self.simulator.run, args=(self, self.vtk_exporter))

        self.simulator.reinit()
        self.thread.start()
        self.progress.description = "running"
        self.viewer.reinit(self.simulator.products)

    def _handle_save(self, _):
        def task(controller):
            controller.progress.value = 0
            if self.checkbox.value:
                self.vtk_exporter.write_pvd()
                controller.progress.description = "VTK..."
                shutil.make_archive(
                    self.vtk_file.absolute_path[:-4], "zip", self.vtk_exporter.path
                )
            controller.progress.description = "netCDF..."
            self.ncdf_exporter.run(controller)
            controller.link.children = (
                (self.ncdf_file.make_link_widget(),)
                if not self.checkbox.value
                else (
                    self.ncdf_file.make_link_widget(),
                    self.vtk_file.make_link_widget(),
                )
            )

        self.thread = Thread(target=task, args=(self,))
        self.thread.start()
        self._setup_stop()

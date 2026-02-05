#!/usr/bin/env pvpython
"""
ParaView pvpython script for visualizing sd_products and sd_attributes.
"""

import argparse
from collections import namedtuple
import pathlib
from paraview import simple as pvs  # pylint: disable=import-error

pvs._DisableFirstRenderCameraReset()  # pylint: disable=protected-access


def cli_using_argparse(argparse_parser):
    """
    Command line interface using argparse.
    """
    argparse_parser.add_argument(
        "--sd-products-pvd",
        dest="sd_products_pvd",
        help="Path to sd_products.pvd",
    )
    argparse_parser.add_argument(
        "--sd-attributes-pvd",
        dest="sd_attributes_pvd",
        help="Path to sd_attributes.pvd",
    )
    argparse_parser.add_argument(
        "--output-animation-path",
        help="Output path for the animation file.",
    )
    argparse_parser.add_argument(
        "--output-screenshot-path",
        help="Output path for the screenshot file.",
    )

    argparse_parser.add_argument(
        "--scalarbar-title-size",
        type=int,
        default=30,
        help="Font size for scalar bar titles.",
    )
    argparse_parser.add_argument(
        "--scalarbar-label-size",
        type=int,
        default=30,
        help="Font size for scalar bar labels.",
    )
    argparse_parser.add_argument(
        "--axes-title-size",
        type=int,
        default=30,
        help="Font size for axes titles.",
    )
    argparse_parser.add_argument(
        "--axes-label-size",
        type=int,
        default=30,
        help="Font size for axes labels.",
    )


parser = argparse.ArgumentParser(
    description="ParaView pvpython script for visualizing sd_products and sd_attributes PVD files."
)
cli_using_argparse(parser)
args = parser.parse_args()

sd_productspvd = pvs.PVDReader(
    registrationName="sd_products.pvd", FileName=args.sd_products_pvd
)
sd_attributespvd = pvs.PVDReader(
    registrationName="sd_attributes.pvd", FileName=args.sd_attributes_pvd
)

setup = {
    "renderView1": pvs.GetActiveViewOrCreate("RenderView"),
    "sd_productspvdDisplay": pvs.Show(
        sd_productspvd,
        pvs.GetActiveViewOrCreate("RenderView"),
        "UnstructuredGridRepresentation",
    ),
    "sd_attributespvdDisplay": pvs.Show(
        sd_attributespvd,
        pvs.GetActiveViewOrCreate("RenderView"),
        "UnstructuredGridRepresentation",
    ),
    "rHlookup_table": pvs.GetColorTransferFunction("RH"),
    "volumelookup_table": pvs.GetColorTransferFunction("volume"),
}

animation_setup = namedtuple("setup", setup.keys())(**setup)


def configure_color_bar_and_display(
    display, lookup_table, kind, *, anim_setup=animation_setup
):
    """
    Configure color bar and display settings.
    """
    display.Representation = "Surface"
    display.SetScalarBarVisibility(anim_setup.renderView1, True)
    color_bar = pvs.GetScalarBar(lookup_table, anim_setup.renderView1)
    color_bar.LabelColor = [0.0, 0.0, 0.0]
    color_bar.DrawScalarBarOutline = 1
    color_bar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
    color_bar.TitleColor = [0.0, 0.0, 0.0]

    color_bar.TitleFontSize = args.scalarbar_title_size
    color_bar.LabelFontSize = args.scalarbar_label_size

    if kind == "prod":
        display.Opacity = 0.4
        display.DisableLighting = 1
        display.Diffuse = 0.76
        lookup_table.RescaleTransferFunction(90.0, 101.0)
        lookup_table.ApplyPreset("Black, Blue and White", True)
        lookup_table.NanColor = [0.67, 1.0, 1.0]
    else:
        display.PointSize = 13.0
        display.RenderPointsAsSpheres = 1
        display.Interpolation = "PBR"
        lookup_table.RescaleTransferFunction(1e-18, 1e-13)
        lookup_table.ApplyPreset("Cold and Hot", True)
        lookup_table.MapControlPointsToLogSpace()
        lookup_table.UseLogScale = 1
        lookup_table.NumberOfTableValues = 16
        lookup_table.InvertTransferFunction()


def configure_data_axes_grid(display, kind):
    """
    Configure data axes grid settings.
    """
    display.DataAxesGrid.GridAxesVisibility = 1
    display.DataAxesGrid.XTitle = ""
    display.DataAxesGrid.YTitle = ""
    display.DataAxesGrid.XAxisUseCustomLabels = 1
    display.DataAxesGrid.YAxisUseCustomLabels = 1

    display.DataAxesGrid.XTitleFontSize = args.axes_title_size
    display.DataAxesGrid.XLabelFontSize = args.axes_label_size
    display.DataAxesGrid.YTitleFontSize = args.axes_title_size
    display.DataAxesGrid.YLabelFontSize = args.axes_label_size

    if kind == "prod":
        display.DataAxesGrid.ZTitle = ""
        display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
        display.DataAxesGrid.ShowGrid = 1
        display.DataAxesGrid.LabelUniqueEdgesOnly = 0
        display.DataAxesGrid.ZAxisUseCustomLabels = 1
        display.DataAxesGrid.ZTitleFontSize = args.axes_title_size
        display.DataAxesGrid.ZLabelFontSize = args.axes_label_size
    else:
        display.DataAxesGrid.ZTitle = "     Z [m]     "
        display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
        display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
        display.DataAxesGrid.ZTitleFontSize = args.axes_title_size
        display.DataAxesGrid.ZLabelFontSize = args.axes_label_size


def configure_view_appearance(render_view):
    """
    Configure view appearance settings."""
    render_view.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
    render_view.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
    render_view.OrientationAxesXVisibility = 0
    render_view.OrientationAxesYVisibility = 0
    render_view.OrientationAxesZColor = [0.0, 0.0, 0.0]
    render_view.UseColorPaletteForBackground = 0
    render_view.Background = [1.0, 1.0, 1.0]


def set_camera_view(render_view):
    """
    Set camera view settings.
    """
    layout1 = pvs.GetLayout()
    layout1.SetSize(1592, 1128)
    render_view.CameraPosition = [
        1548.95,
        -1349.49,
        699.27,
    ]
    render_view.CameraFocalPoint = [
        -1.37e-13,
        3.18e-13,
        505.00,
    ]
    render_view.CameraViewUp = [
        -0.07,
        0.06,
        0.99,
    ]
    render_view.CameraParallelScale = 534.08


def save_animation_and_screenshot(render_view, animation_path, screenshot_path):
    """
    Save animation and screenshot.
    """
    animation_scene = pvs.GetAnimationScene()
    animation_scene.UpdateAnimationUsingDataTimeSteps()

    pvs.SaveAnimation(animation_path, render_view, FrameRate=15)

    if not screenshot_path:
        return

    time_steps = sd_productspvd.TimestepValues
    if not time_steps:
        return

    last_time = time_steps[-1]
    render_view.ViewTime = last_time

    for reader in (sd_productspvd, sd_attributespvd):
        reader.UpdatePipeline(last_time)

    pvs.ExportView(
        filename=str(pathlib.Path(screenshot_path)),
        view=render_view,
        Rasterize3Dgeometry=False,
        GL2PSdepthsortmethod="BSP sorting (slow, best)",
    )

    pvs.RenderAllViews()


configure_color_bar_and_display(
    animation_setup.sd_productspvdDisplay, animation_setup.rHlookup_table, kind="prod"
)
configure_data_axes_grid(animation_setup.sd_productspvdDisplay, kind="prod")
configure_view_appearance(animation_setup.renderView1)
configure_color_bar_and_display(
    animation_setup.sd_attributespvdDisplay,
    animation_setup.volumelookup_table,
    kind="attr",
)
configure_data_axes_grid(animation_setup.sd_attributespvdDisplay, kind="attr")
set_camera_view(animation_setup.renderView1)
save_animation_and_screenshot(
    animation_setup.renderView1, args.output_animation_path, args.output_screenshot_path
)

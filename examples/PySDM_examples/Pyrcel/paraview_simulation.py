"""
ParaView pvpython script for visualizing sd_products and sd_attributes.
"""

import argparse
from paraview import simple as pvs  # pylint: disable=import-error


def cli_using_argparse(argparse_parser):
    """
    Command line interface using argparse.
    """
    argparse_parser.add_argument(
        "--sd-products-pvd",
        dest="sd_products_pvd",
        default=r"C:\\Users\\strza\\Desktop\\PySDM\\examples\\PySDM_examples\\Pyrcel\\output\\sd_products.pvd",  # pylint: disable=line-too-long
        help="Path to sd_products.pvd",
    )
    argparse_parser.add_argument(
        "--sd-attributes-pvd",
        dest="sd_attributes_pvd",
        default=r"C:\\Users\\strza\\Desktop\\PySDM\\examples\\PySDM_examples\\Pyrcel\\output\\sd_attributes.pvd",  # pylint: disable=line-too-long
        help="Path to sd_attributes.pvd",
    )


parser = argparse.ArgumentParser(
    description="ParaView pvpython script for visualizing sd_products and sd_attributes PVD files."
)
cli_using_argparse(parser)
args = parser.parse_args()

# TODO tuple
pvs._DisableFirstRenderCameraReset()  # pylint: disable=protected-access
renderView1 = pvs.GetActiveViewOrCreate("RenderView")
sd_productspvd = pvs.PVDReader(
    registrationName="sd_products.pvd",
    FileName=args.sd_products_pvd,
)
sd_productspvdDisplay = pvs.Show(
    sd_productspvd, renderView1, "UnstructuredGridRepresentation"
)
sd_attributespvd = pvs.PVDReader(
    registrationName="sd_attributes.pvd",
    FileName=args.sd_attributes_pvd,
)
sd_attributespvdDisplay = pvs.Show(
    sd_attributespvd, renderView1, "UnstructuredGridRepresentation"
)
rHlookup_table = pvs.GetColorTransferFunction("RH")
volumelookup_table = pvs.GetColorTransferFunction("volume")


def configure_color_bar_and_display(display, lookup_table, kind):
    """
    Configure color bar and display settings.
    """
    display.Representation = "Surface"
    display.SetScalarBarVisibility(renderView1, True)
    color_bar = pvs.GetScalarBar(lookup_table, renderView1)
    color_bar.LabelColor = [0.0, 0.0, 0.0]
    color_bar.DrawScalarBarOutline = 1
    color_bar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
    color_bar.TitleColor = [0.0, 0.0, 0.0]

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

    if kind == "prod":
        display.DataAxesGrid.ZTitle = ""
        display.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
        display.DataAxesGrid.ShowGrid = 1
        display.DataAxesGrid.LabelUniqueEdgesOnly = 0
        display.DataAxesGrid.ZAxisUseCustomLabels = 1
    else:
        display.DataAxesGrid.ZTitle = "Z [m]"
        display.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
        display.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]


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
        1548.945972263949,
        -1349.493616194682,
        699.2699178747185,
    ]
    render_view.CameraFocalPoint = [
        -1.3686701146742275e-13,
        3.1755031232028544e-13,
        505.00000000000017,
    ]
    render_view.CameraViewUp = [
        -0.07221292769632315,
        0.06043215330151439,
        0.9955567527373153,
    ]
    render_view.CameraParallelScale = 534.07781536883


def save_animation_and_screenshot(render_view, animation_path, screenshot_path):
    """
    Save animation and screenshot.
    """
    animation_scene = pvs.GetAnimationScene()
    animation_scene.UpdateAnimationUsingDataTimeSteps()

    pvs.SaveAnimation(animation_path, render_view, FrameRate=15)
    pvs.SaveScreenshot(screenshot_path, render_view)


configure_color_bar_and_display(sd_productspvdDisplay, rHlookup_table, kind="prod")
configure_data_axes_grid(sd_productspvdDisplay, kind="prod")
configure_view_appearance(renderView1)
configure_color_bar_and_display(
    sd_attributespvdDisplay, volumelookup_table, kind="attr"
)
configure_data_axes_grid(sd_attributespvdDisplay, kind="attr")
set_camera_view(renderView1)
OUTPUT_ANIMATION_PATH = r"C:\Users\strza\Desktop\PySDM\examples\PySDM_examples\Pyrcel\output_animation.avi"  # pylint: disable=line-too-long
OUTPUT_SCREENSHOT_PATH = (
    r"C:\Users\strza\Desktop\PySDM\examples\PySDM_examples\Pyrcel\last_frame.png"
)
save_animation_and_screenshot(
    renderView1, OUTPUT_ANIMATION_PATH, OUTPUT_SCREENSHOT_PATH
)

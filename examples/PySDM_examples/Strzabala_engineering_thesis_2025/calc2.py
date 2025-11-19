"""Module for processing PVD files and rendering visualizations using ParaView."""

#!/usr/bin/env pvpython
import argparse
from collections import namedtuple
import pathlib

from paraview import simple as pvs  # pylint: disable=import-error

pvs._DisableFirstRenderCameraReset()  # pylint: disable=protected-access


def cli_using_argparse(argp):
    """Set up command line argument parsing."""
    argp.add_argument("product_path", help="path to pvd products file")
    argp.add_argument("attributes_path", help=" path to pvd attributes file")
    argp.add_argument("output_path", help="path where to write output files")
    argp.add_argument(
        "--mode",
        choices=["light", "dark"],
        default="light",
        help="Choose 'light' or 'dark' mode.",
    )
    argp.add_argument(
        "--multiplicity_preset",
        default="Inferno (matplotlib)",
        help="Preset for multiplicity",
    )
    argp.add_argument(
        "--multiplicity_logscale",
        action="store_false",
        help="Use log scale for multiplicity",
    )
    argp.add_argument(
        "--effectiveradius_preset",
        default="Black, Blue and White",
        help="Preset for effectiveradius",
    )
    argp.add_argument(
        "--effectiveradius_logscale",
        action="store_false",
        help="Use log scale for effectiveradius",
    )
    argp.add_argument(
        "--effectiveradius_nan_color",
        nargs=3,
        type=float,
        default=[0.666, 0.333, 1.0],
        help="Nan color in RGB format for effectiveradius",
    )
    argp.add_argument(
        "--sd_products_opacity", type=float, default=0.0, help="Opacity for sd_products"
    )
    argp.add_argument(
        "--calculator1_opacity",
        type=float,
        default=0.0,
        help="Opacity for calculator1",
    )
    argp.add_argument(
        "--sd_attributes_opacity",
        type=float,
        default=0.0,
        help="Opacity for sd_attributes",
    )
    argp.add_argument(
        "--animation_size",
        nargs=2,
        type=int,
        default=[800, 800],
        help="Animation size [x,y]",
    )
    argp.add_argument(
        "--animationframename",
        type=str,
        help="Name of the file with animation last frame",
    )
    argp.add_argument(
        "--animationname",
        type=str,
        help="Name of the file with animation",
    )
    argp.add_argument(
        "--framerate", type=int, help="Number of frame rates.", default=15
    )


ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
cli_using_argparse(ap)

args = ap.parse_args()
sd_productspvd = pvs.OpenDataFile(args.product_path)
sd_attributespvd = pvs.OpenDataFile(args.attributes_path)


setup = {
    "renderView1": pvs.GetActiveViewOrCreate("RenderView"),
    "sd_attributespvdDisplay": pvs.GetDisplayProperties(
        sd_attributespvd, view=pvs.GetActiveViewOrCreate("RenderView")
    ),
    "effectiveradiusLUT": pvs.GetColorTransferFunction("effectiveradius"),
    "sd_productspvdDisplay": pvs.GetDisplayProperties(
        sd_productspvd, view=pvs.GetActiveViewOrCreate("RenderView")
    ),
    "color": [1, 1, 1] if args.mode == "dark" else [0, 0, 0],
    "inverted_color": [0.129, 0.145, 0.161] if args.mode == "dark" else [1, 1, 1],
}

setup = namedtuple("Setup", setup.keys())(**setup)

setup.effectiveradiusLUT.RescaleTransferFunction(0.1380997175392798, 207.7063518856934)
materialLibrary1 = pvs.GetMaterialLibrary()
setup.renderView1.Update()


def create_new_calculator(params):
    """Create a new calculator in ParaView."""
    calcinput = params["calcinput"]
    representation = params["representation"]
    function = params["function"]
    color_by1 = params["color_by1"]
    color_by2 = params["color_by2"]
    color_by3 = params["color_by3"]
    scalar_coloring = params.get("scalar_coloring", False)
    hide = params.get("hide", False)
    setup_context = params["setup_context"]
    registration_name = params["registration_name"]

    calculator = pvs.Calculator(registrationName=registration_name, Input=calcinput)
    display = pvs.Show(calculator, setup_context.renderView1, representation)
    calculator.Function = function
    setup_context.renderView1.Update()
    if scalar_coloring:
        pvs.ColorBy(display, (color_by1, color_by2, color_by3))
    if hide:
        pvs.Hide(calculator, setup_context.renderView1)
    return setup_context.renderView1.Update()


def scalar_bar(name, *, y, erLUT):
    """Display a scalar bar for the given name."""
    calculator1_display = pvs.Show(
        calculator1, y.renderView1, "UnstructuredGridRepresentation"
    )
    calculator1_display.SetScalarBarVisibility(y.renderView1, True)
    scalar_bar_instance = pvs.GetScalarBar(erLUT.effectiveradiusLUT, y.renderView1)
    scalar_bar_instance.ComponentTitle = ""
    scalar_bar_instance.Title = name
    scalar_bar_instance.TitleFontSize = 25
    scalar_bar_instance.LabelFontSize = 25
    scalar_bar_instance.LabelColor = setup.color
    scalar_bar_instance.TitleColor = setup.color
    pvs.Hide(scalar_bar_instance)
    y.renderView1.Update()


def create_glyph(params):
    """Create a glyph representation in ParaView."""
    registration_name = params["registration_name"]
    put = params["put"]
    scale_array1 = params["scale_array1"]
    scale_array2 = params["scale_array2"]
    color_by = params.get("color_by", False)

    glyph = pvs.Glyph(registrationName=registration_name, Input=put, GlyphType="Arrow")
    glyph_display = pvs.Show(glyph, setup.renderView1, "GeometryRepresentation")
    glyph_display.Representation = "Surface"
    glyph.ScaleArray = [scale_array1, scale_array2]
    glyph.ScaleFactor = 100
    glyph_display.SetScalarBarVisibility(setup.renderView1, True)

    multiplicity_lut = pvs.GetColorTransferFunction("multiplicity")
    multiplicity_lut_color_bar = pvs.GetScalarBar(multiplicity_lut, setup.renderView1)
    multiplicity_lut_color_bar.Position = [0.5, 0.9]
    multiplicity_lut_color_bar.TitleFontSize = 25
    multiplicity_lut_color_bar.LabelFontSize = 25
    multiplicity_lut_color_bar.LabelColor = setup.color
    multiplicity_lut_color_bar.TitleColor = setup.color

    if color_by:
        glyph_display.ColorArrayName = ["POINTS", ""]
        pvs.ColorBy(glyph_display, None)

    setup.renderView1.Update()


def apply_presets_logscale_opacity_and_update(*, y, attdisplay, erLUT, proddisplay):
    """Apply presets, log scale, opacity settings, and update the view."""
    multiplicity_lut = pvs.GetColorTransferFunction("multiplicity")
    multiplicity_lut.RescaleTransferFunction(19951.0, 50461190157.0)
    calculator1_display = pvs.Show(
        calculator1, y.renderView1, "UnstructuredGridRepresentation"
    )
    multiplicity_lut.ApplyPreset(args.multiplicity_preset, True)
    if args.multiplicity_logscale:
        multiplicity_lut.MapControlPointsToLogSpace()
        multiplicity_lut.UseLogScale = 1
    else:
        multiplicity_lut.MapControlPointsToLinearSpace()
        multiplicity_lut.UseLogScale = 0

    erLUT.effectiveradiusLUT.ApplyPreset(args.effectiveradius_preset, True)
    if args.effectiveradius_logscale:
        erLUT.effectiveradiusLUT.MapControlPointsToLogSpace()
        erLUT.effectiveradiusLUT.UseLogScale = 1
    else:
        erLUT.effectiveradiusLUT.MapControlPointsToLinearSpace()
        erLUT.effectiveradiusLUT.UseLogScale = 0

    erLUT.effectiveradiusLUT.NanColor = args.effectiveradius_nan_color
    proddisplay.sd_productspvdDisplay.Opacity = args.sd_products_opacity
    proddisplay.sd_productspvdDisplay.SetScalarBarVisibility(y.renderView1, False)
    calculator1_display.Opacity = args.calculator1_opacity
    attdisplay.sd_attributespvdDisplay.Opacity = args.sd_attributes_opacity
    y.renderView1.Update()


def get_layout(*, y):
    """Set the layout for the render view."""
    pvs.SetViewProperties(
        Background=setup.inverted_color, UseColorPaletteForBackground=0
    )
    pvs.Render(setup.renderView1)
    layout1 = pvs.GetLayout()
    layout1.SetSize(args.animation_size)
    layout1.PreviewMode = args.animation_size
    y.renderView1.Update()


def set_current_camera_placement(*, y):
    """Set the camera placement for the render view."""
    y.renderView1.InteractionMode = "2D"
    y.renderView1.CameraPosition = [836, 677, -4098]
    y.renderView1.CameraFocalPoint = [636, 1030, 0.0]
    y.renderView1.CameraViewUp = [1.0, 0.0, 0.0]
    y.renderView1.CameraParallelScale = 1560
    y.renderView1.Update()


def axes_settings(*, view):
    """Configure axes settings for the render view."""
    view.CenterAxesVisibility = True
    view.OrientationAxesVisibility = False
    axes_grid = view.AxesGrid
    axes_grid.Visibility = True
    axes_grid.XTitle = "Z [m]"
    axes_grid.YTitle = "X [m]"

    axes_grid.XAxisUseCustomLabels = True
    axes_grid.XAxisLabels = [300, 600, 900, 1200]
    axes_grid.YAxisUseCustomLabels = True
    axes_grid.YAxisLabels = [300, 600, 900, 1200]

    axes_grid.XTitleFontSize = 30
    axes_grid.XLabelFontSize = 30
    axes_grid.YTitleFontSize = 30
    axes_grid.YLabelFontSize = 30

    axes_grid.XTitleColor = setup.color
    axes_grid.XLabelColor = setup.color
    axes_grid.YTitleColor = setup.color
    axes_grid.YLabelColor = setup.color
    axes_grid.GridColor = [0.1, 0.1, 0.1]
    view.CenterAxesVisibility = False
    view.Update()


def time_annotation(*, y):
    """Add a time annotation to the render view."""
    time = pvs.AnnotateTimeFilter(
        guiName="AnnotateTimeFilter1", Scale=1 / 60, Format="Time:{time:g}min"
    )
    timedisplay = pvs.Show(time, y.renderView1)
    timedisplay.FontSize = 25
    timedisplay.WindowLocation = "Any Location"
    timedisplay.FontSize = 30
    timedisplay.Position = [0.4, 0.9]
    timedisplay.Color = setup.color
    y.renderView1.Update()


def text(text_in, position_y, *, view):
    """Display text in the render view at a specified position."""
    sentence = pvs.Text()
    sentence.Text = text_in
    text_display = pvs.Show(sentence, view)
    text_display.Color = setup.color
    text_display.WindowLocation = "Any Location"
    text_display.FontSize = 28
    text_display.Position = [0.17, position_y]


def last_anim_frame(animation_frame_name):
    """Export the last animation frame to a file."""
    time_steps = sd_productspvd.TimestepValues
    last_time = time_steps[90]
    setup.renderView1.ViewTime = last_time
    for reader in (sd_productspvd,):
        reader.UpdatePipeline(last_time)
        pvs.ExportView(
            filename=str(pathlib.Path(args.output_path) / animation_frame_name),
            view=setup.renderView1,
            Rasterize3Dgeometry=False,
            GL2PSdepthsortmethod="BSP sorting (slow, best)",
        )
    pvs.RenderAllViews()


calculator1 = create_new_calculator(
    {
        "calcinput": sd_attributespvd,
        "representation": "UnstructuredGridRepresentation",
        "function": '"relative fall velocity"*(-iHat)',
        "color_by1": "None",
        "color_by2": "None",
        "color_by3": "None",
        "scalar_coloring": False,
        "hide": False,
        "setup_context": setup,
        "registration_name": "Calculator1",
    }
)

create_glyph(
    {
        "registration_name": "Glyph1",
        "put": calculator1,
        "scale_array1": "POINTS",
        "scale_array2": "relative fall velocity",
        "color_by": False,
        "setup_context": setup,
    }
)

apply_presets_logscale_opacity_and_update(
    y=setup, attdisplay=setup, erLUT=setup, proddisplay=setup
)
get_layout(y=setup)
set_current_camera_placement(y=setup)
axes_settings(view=setup.renderView1)
time_annotation(y=setup)
if args.animationframename is not None:
    last_anim_frame(animation_frame_name=args.animationframename)
scene = pvs.GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
pvs.Render(setup.renderView1)
pvs.SaveAnimation(
    str(pathlib.Path(args.output_path) / args.animationname),
    setup.renderView1,
    FrameRate=args.framerate,
)
pvs.RenderAllViews()

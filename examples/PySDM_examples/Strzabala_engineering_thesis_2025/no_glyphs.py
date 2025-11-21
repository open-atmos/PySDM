#!/usr/bin/env pvpython
"""Paraview rendering script.

This module prepares Paraview scene, calculators, glyphs and exports animation/images.
Lint fixes: module docstring, protected-access disabled inline, reduced function args,
and short docstrings added to public functions.
"""
import argparse
from collections import namedtuple
import pathlib

from paraview import simple as pvs  # pylint: disable=import-error

pvs._DisableFirstRenderCameraReset()  # pylint: disable=protected-access


def cli_using_argparse(argp):
    """Add and document command-line arguments used by this pvpython script.

    Parameters
    ----------
    argp : argparse.ArgumentParser
        ArgumentParser instance to which this function will add arguments.
    """
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
        "--sd_products_opacity", type=float, default=0.9, help="Opacity for sd_products"
    )
    argp.add_argument(
        "--calculator1_opacity",
        type=float,
        default=0.19,
        help="Opacity for calculator1",
    )
    argp.add_argument(
        "--sd_attributes_opacity",
        type=float,
        default=0.77,
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


def create_new_calculator(
    calcinput,
    representation,
    function,
    color_by,
    *,
    scalar_coloring=False,
    hide=False,
    y,
    registration_name,
):
    """Create and show a Calculator filter; return the calculator object.

    color_by should be a tuple (arrayName, association, component) or None.
    """
    calculator = pvs.Calculator(registrationName=registration_name, Input=calcinput)
    display = pvs.Show(calculator, y.renderView1, representation)
    calculator.Function = function
    y.renderView1.Update()
    if scalar_coloring and color_by:
        pvs.ColorBy(display, tuple(color_by))
    if hide:
        pvs.Hide(calculator, y.renderView1)
    return calculator


def scalar_bar(name, *, y, erLUT, calculator):
    """Ensure scalar bar for given calculator is visible and styled."""
    calculator_display = pvs.Show(
        calculator, y.renderView1, "UnstructuredGridRepresentation"
    )
    calculator_display.SetScalarBarVisibility(y.renderView1, True)
    scalar_bar_obj = pvs.GetScalarBar(erLUT.effectiveradiusLUT, y.renderView1)
    scalar_bar_obj.ComponentTitle = ""
    scalar_bar_obj.Title = name
    scalar_bar_obj.TitleFontSize = 25
    scalar_bar_obj.LabelFontSize = 25
    scalar_bar_obj.LabelColor = setup.color
    scalar_bar_obj.TitleColor = setup.color
    y.renderView1.Update()


def create_glyph(
    registration_name, put, scale_array1, scale_array2, color_by=False, *, y
):
    """Create a glyph and return it."""
    glyph = pvs.Glyph(registrationName=registration_name, Input=put, GlyphType="Arrow")
    glyph_display = pvs.Show(glyph, y.renderView1, "GeometryRepresentation")
    glyph_display.Representation = "Surface"
    glyph.ScaleArray = [scale_array1, scale_array2]
    glyph.ScaleFactor = 100
    glyph_display.SetScalarBarVisibility(y.renderView1, True)
    multiplicity_lut = pvs.GetColorTransferFunction("multiplicity")
    multiplicity_lut_color_bar = pvs.GetScalarBar(multiplicity_lut, y.renderView1)
    multiplicity_lut_color_bar.TitleFontSize = 25
    multiplicity_lut_color_bar.LabelFontSize = 25
    multiplicity_lut_color_bar.LabelColor = setup.color
    multiplicity_lut_color_bar.TitleColor = setup.color
    if color_by:
        glyph_display.ColorArrayName = ["POINTS", ""]
        pvs.ColorBy(glyph_display, None)
    y.renderView1.Update()
    return glyph


def apply_presets_logscale_opacity_and_update(*, y, attdisplay, erLUT, proddisplay):
    """Apply LUT presets / logscale and set object opacities."""
    multiplicity_lut = pvs.GetColorTransferFunction("multiplicity")
    multiplicity_lut.RescaleTransferFunction(19951.0, 50461190157.0)
    calculator_1_display = pvs.Show(
        calculator_1, y.renderView1, "UnstructuredGridRepresentation"
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

    proddisplay.sd_productspvdDisplay.SetRepresentationType("Surface With Edges")
    proddisplay.sd_productspvdDisplay.Opacity = args.sd_products_opacity
    calculator_1_display.Opacity = args.calculator1_opacity
    attdisplay.sd_attributespvdDisplay.Opacity = args.sd_attributes_opacity

    y.renderView1.Update()


def get_layout(*, y):
    """Configure layout size for animation rendering."""
    pvs.SetViewProperties(
        Background=setup.inverted_color, UseColorPaletteForBackground=0
    )
    pvs.Render(setup.renderView1)
    layout1 = pvs.GetLayout()
    layout1.SetSize(args.animation_size)
    layout1.PreviewMode = args.animation_size
    y.renderView1.Update()


def set_current_camera_placement(*, y):
    """Set camera placement for the scene."""
    y.renderView1.InteractionMode = "2D"
    y.renderView1.CameraPosition = [
        836,
        677,
        -4098,
    ]
    y.renderView1.CameraFocalPoint = [636, 1030, 0.0]
    y.renderView1.CameraViewUp = [1.0, 0.0, 0.0]
    y.renderView1.CameraParallelScale = 1560
    y.renderView1.Update()


def axes_settings(*, view):
    """Configure axes grid, labels and styling on given view."""
    # setup.renderView1.Background = [1,0.5,0.2]
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
    """Add time annotation to the view."""
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
    """Place a small text label on view (no return)."""
    sentence = pvs.Text()
    sentence.Text = text_in
    text_display = pvs.Show(sentence, view)
    text_display.Color = setup.color
    text_display.WindowLocation = "Any Location"
    text_display.FontSize = 28
    text_display.Position = [0.17, position_y]


def last_anim_frame(animation_frame_name):
    """Export last animation frame to a file in output path."""
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


calculator_1 = create_new_calculator(
    sd_attributespvd,
    "UnstructuredGridRepresentation",
    '"relative fall velocity"*(-iHat)',
    ("None", "None", "None"),
    y=setup,
    registration_name="Calculator1",
)
scalar_bar("effective radius [um]", y=setup, erLUT=setup, calculator=calculator_1)
glyph_1 = create_glyph(
    "Glyph1", calculator_1, "POINTS", "relative fall velocity", y=setup
)
pvs.Hide(glyph_1)
calculator_2 = create_new_calculator(
    sd_productspvd,
    "StructuredGridRepresentation",
    "cx*jHat+cy*iHat",
    ("CELLS", "Result", "Magnitude"),
    scalar_coloring=True,
    hide=True,
    y=setup,
    registration_name="Calculator2",
)
apply_presets_logscale_opacity_and_update(
    y=setup, attdisplay=setup, erLUT=setup, proddisplay=setup
)
glyph_2 = create_glyph("Glyph2", calculator_2, "CELLS", "Result", True, y=setup)
pvs.Hide(glyph_2)
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

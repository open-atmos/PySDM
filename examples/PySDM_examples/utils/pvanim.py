#!/usr/bin/env pvpython
import argparse
from collections import namedtuple
import pathlib

from paraview import simple as pvs  # pylint: disable=import-error

pvs._DisableFirstRenderCameraReset()


def cli_using_argparse(argp):
    argp.add_argument("product_path", help="path to pvd products file")
    argp.add_argument("attributes_path", help=" path to pvd attributes file")
    argp.add_argument("output_path", help="path where to write output files")
    argp.add_argument(
        "--mode",
        choices=["light", "dark"],
        default="dark",
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
    color_by1,
    color_by2,
    color_by3,
    scalar_coloring=False,
    hide=False,
    *,
    y,
    registrationame,
):
    calculator = pvs.Calculator(registrationName=registrationame, Input=calcinput)
    display = pvs.Show(calculator, y.renderView1, representation)
    calculator.Function = function
    y.renderView1.Update()
    if scalar_coloring is True:
        pvs.ColorBy(display, (color_by1, color_by2, color_by3))
    if hide is True:
        pvs.Hide(calculator, y.renderView1)
    return y.renderView1.Update()


def scalar_bar(name, *, y, erLUT):
    calculator1Display = pvs.Show(
        calculator1, y.renderView1, "UnstructuredGridRepresentation"
    )
    calculator1Display.SetScalarBarVisibility(y.renderView1, True)
    scalarBar = pvs.GetScalarBar(erLUT.effectiveradiusLUT, y.renderView1)
    scalarBar.ComponentTitle = ""
    scalarBar.Title = name
    scalarBar.TitleFontSize = 25
    scalarBar.LabelFontSize = 25
    scalarBar.LabelColor = setup.color
    scalarBar.TitleColor = setup.color
    y.renderView1.Update()


def create_glyph(
    registration_name, put, scale_array1, scale_array2, color_by=False, *, y
):
    glyph = pvs.Glyph(registrationName=registration_name, Input=put, GlyphType="Arrow")
    glyphDisplay = pvs.Show(glyph, y.renderView1, "GeometryRepresentation")
    glyphDisplay.Representation = "Surface"
    glyph.ScaleArray = [scale_array1, scale_array2]
    glyph.ScaleFactor = 100
    glyphDisplay.SetScalarBarVisibility(y.renderView1, True)
    multiplicityLUT = pvs.GetColorTransferFunction("multiplicity")
    multiplicityLUTColorBar = pvs.GetScalarBar(multiplicityLUT, y.renderView1)
    multiplicityLUTColorBar.TitleFontSize = 25
    multiplicityLUTColorBar.LabelFontSize = 25
    multiplicityLUTColorBar.LabelColor = setup.color
    multiplicityLUTColorBar.TitleColor = setup.color
    if color_by is True:
        glyphDisplay.ColorArrayName = ["POINTS", ""]
        pvs.ColorBy(glyphDisplay, None)
    y.renderView1.Update()


def apply_presets_logscale_opacity_and_update(*, y, attdisplay, erLUT, proddisplay):
    multiplicityLUT = pvs.GetColorTransferFunction("multiplicity")
    multiplicityLUT.RescaleTransferFunction(19951.0, 50461190157.0)
    calculator1Display = pvs.Show(
        calculator1, y.renderView1, "UnstructuredGridRepresentation"
    )
    multiplicityLUT.ApplyPreset(args.multiplicity_preset, True)
    if args.multiplicity_logscale:
        multiplicityLUT.MapControlPointsToLogSpace()
        multiplicityLUT.UseLogScale = 1
    else:
        multiplicityLUT.MapControlPointsToLinearSpace()
        multiplicityLUT.UseLogScale = 0

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
    calculator1Display.Opacity = args.calculator1_opacity
    attdisplay.sd_attributespvdDisplay.Opacity = args.sd_attributes_opacity

    y.renderView1.Update()


def get_layout(*, y):
    pvs.SetViewProperties(
        Background=setup.inverted_color, UseColorPaletteForBackground=0
    )
    pvs.Render(setup.renderView1)
    layout1 = pvs.GetLayout()
    layout1.SetSize(args.animation_size)
    layout1.PreviewMode = args.animation_size
    y.renderView1.Update()


def set_current_camera_placement(*, y):
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
    # setup.renderView1.Background = [1,0.5,0.2]
    view.CenterAxesVisibility = True
    view.OrientationAxesVisibility = False
    axesGrid = view.AxesGrid
    axesGrid.Visibility = True
    axesGrid.XTitle = "Z [m]"
    axesGrid.YTitle = "X [m]"

    axesGrid.XAxisUseCustomLabels = True
    axesGrid.XAxisLabels = [300, 600, 900, 1200]
    axesGrid.YAxisUseCustomLabels = True
    axesGrid.YAxisLabels = [300, 600, 900, 1200]

    axesGrid.XTitleFontSize = 30
    axesGrid.XLabelFontSize = 30
    axesGrid.YTitleFontSize = 30
    axesGrid.YLabelFontSize = 30

    axesGrid.XTitleColor = setup.color
    axesGrid.XLabelColor = setup.color
    axesGrid.YTitleColor = setup.color
    axesGrid.YLabelColor = setup.color
    axesGrid.GridColor = [0.1, 0.1, 0.1]
    view.CenterAxesVisibility = False
    view.Update()


def time_annotation(*, y):
    time = pvs.AnnotateTimeFilter(
        guiName="AnnotateTimeFilter1", Scale=1 / 60, Format="Time:{time:g}min"
    )
    timedisplay = pvs.Show(time, y.renderView1)
    timedisplay.FontSize = 25
    timedisplay.WindowLocation = "Any Location"
    timedisplay.FontSize = 30
    timedisplay.Position = [0.31, 0.9]
    timedisplay.Color = setup.color
    y.renderView1.Update()


def text(text_in, position_y, *, view):
    sentence = pvs.Text()
    sentence.Text = text_in
    textDisplay = pvs.Show(sentence, view)
    textDisplay.Color = setup.color
    textDisplay.WindowLocation = "Any Location"
    textDisplay.FontSize = 28
    textDisplay.Position = [0.17, position_y]


def last_anim_frame(animation_frame_name):
    time_steps = sd_productspvd.TimestepValues
    last_time = time_steps[len(time_steps) - 1]
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
    sd_attributespvd,
    "UnstructuredGridRepresentation",
    '"relative fall velocity"*(-iHat)',
    "None",
    "None",
    "None",
    y=setup,
    registrationame="Calculator1",
)
scalar_bar("effective radius [um]", y=setup, erLUT=setup)
create_glyph("Glyph1", calculator1, "POINTS", "relative fall velocity", y=setup)
calculator2 = create_new_calculator(
    sd_productspvd,
    "StructuredGridRepresentation",
    "cx*jHat+cy*iHat",
    "CELLS",
    "Result",
    "Magnitude",
    True,
    True,
    y=setup,
    registrationame="Calculator2",
)
apply_presets_logscale_opacity_and_update(
    y=setup, attdisplay=setup, erLUT=setup, proddisplay=setup
)
create_glyph("Glyph2", calculator2, "CELLS", "Result", True, y=setup)
get_layout(y=setup)
set_current_camera_placement(y=setup)
axes_settings(view=setup.renderView1)
time_annotation(y=setup)
text("Arrows scale with Courant", 0.15, view=setup.renderView1)
text("number C=u·Δt/Δx, reflecting", 0.1, view=setup.renderView1)
text("the grid spacing Δx and Δy.", 0.05, view=setup.renderView1)
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

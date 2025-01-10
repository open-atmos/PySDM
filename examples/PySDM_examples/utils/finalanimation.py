from paraview.simple import *
import argparse
from pathlib import Path
paraview.simple._DisableFirstRenderCameraReset()


def command_line_argument_parser_using_argparse(ap):
    ap.add_argument("product_path", help = 'path to pvd products file')
    ap.add_argument("attributes_path", help=' path to pvd attributes file')

    ap.add_argument('--multiplicity_preset', default = 'Inferno (matplotlib)', help = 'Preset for multiplicity')
    ap.add_argument('--multiplicity_logscale', action = 'store_true', help = 'Use log scale for multiplicity')
    ap.add_argument('--effectiveradius_preset', default = 'Black, Blue and White', help = 'Preset for effectiveradius')
    ap.add_argument('--effectiveradius_logscale', action = 'store_true', help = 'Use log scale for effectiveradius')
    ap.add_argument('--effectiveradius_nan_color', nargs = 3, type = float, default = [0.666, 0.333, 1.0], help = 'Nan color in RGB format for effectiveradius')
    ap.add_argument('--sd_products_opacity', type = float, default = 0.9, help = 'Opacity for sd_products')
    ap.add_argument('--calculator1_opacity', type = float, default = 0.19, help = 'Opacity for calculator1')
    ap.add_argument('--sd_attributes_opacity', type = float, default = 0.77, help = 'Opacity for sd_attributes')
ap = argparse.ArgumentParser()
command_line_argument_parser_using_argparse(ap)

args = ap.parse_args()
sd_productspvd = OpenDataFile(args.product_path)
sd_attributespvd = OpenDataFile(args.attributes_path)


# startup settings
multiplicityLUT = GetColorTransferFunction('multiplicity')
renderView1 = GetActiveViewOrCreate('RenderView')
sd_attributespvdDisplay = Show(sd_attributespvd, renderView1, 'UnstructuredGridRepresentation')
sd_attributespvdDisplay = GetDisplayProperties(sd_attributespvd, view=renderView1)
effectiveradiusLUT = GetColorTransferFunction('effectiveradius')
multiplicityLUT.RescaleTransferFunction(19951.0, 50461190157.0)
effectiveradiusLUT.RescaleTransferFunction(0.1380997175392798, 207.7063518856934)
sd_productspvdDisplay = GetDisplayProperties(sd_productspvd, view=renderView1)
sd_productspvdDisplay.SetRepresentationType('Surface With Edges')


def create_new_calculator(registrationame, input, index, representation, function, color_by1, color_by2, color_by3, scalar_coloring = False, hide = False):
    calculator = Calculator(registrationName=registrationame, Input=input)
    variable_name = f"calculator{index}Display"
    display = globals()[variable_name] = Show(calculator, renderView1, representation)
    calculator.Function = function
    renderView1.Update()
    if scalar_coloring == True:
        ColorBy(display, (color_by1, color_by2, color_by3))
    else:
        None
    if hide == True:
        Hide(calculator, renderView1)
    else:
        None

calculator1 = create_new_calculator('Calculator1', sd_attributespvd, 1, 'UnstructuredGridRepresentation', '"relative fall velocity"*(-iHat)', 'None', 'None', 'None')


def color_bar_legend(): 
    calculator1Display.SetScalarBarVisibility(renderView1, True)
    scalarBar = GetScalarBar(effectiveradiusLUT, renderView1)
    scalarBar.ComponentTitle = ""
    scalarBar.Title = "effective radius [um]"
    renderView1.Update()

color_bar_legend()


def create_glyph(registration_name, input, scale_array1, scale_array2, scalarbar = True, color_by = False):
    glyph = Glyph(registrationName=registration_name, Input=input,
        GlyphType='Arrow')
    glyphDisplay = Show(glyph, renderView1, 'GeometryRepresentation')
    glyphDisplay.Representation = 'Surface'
    glyph.ScaleArray = [scale_array1, scale_array2]
    glyph.ScaleFactor = 100
    if scalarbar == True:
        glyphDisplay.SetScalarBarVisibility(renderView1, True)
    else:
        glyphDisplay.SetScalarBarVisibility(renderView1, False)
    if color_by == True:
        ColorBy(glyphDisplay, None)
    else:
        None
    renderView1.Update()

glyph1 = create_glyph('Glyph1', calculator1, 'POINTS', 'relative fall velocity')

calculator2 = create_new_calculator('Calculator2', sd_productspvd, 2, 'StructuredGridRepresentation', 'cx*jHat+cy*iHat', 'CELLS', 'Result', 'Magnitude', True, True)

glyph2 = create_glyph('Glyph2', calculator2, 'CELLS', 'Result', False, True)


def apply_presets_logscale_opacity_and_update():
    multiplicityLUT.ApplyPreset(args.multiplicity_preset, True)
    if args.multiplicity_logscale:
        multiplicityLUT.MapControlPointsToLogSpace()
        multiplicityLUT.UseLogScale = 1
    else:
        multiplicityLUT.MapControlPointsToLinearSpace()
        multiplicityLUT.UseLogScale = 0

    effectiveradiusLUT.ApplyPreset(args.effectiveradius_preset, True)
    if args.effectiveradius_logscale:
        effectiveradiusLUT.MapControlPointsToLogSpace()
        effectiveradiusLUT.UseLogScale = 1
    else:
        effectiveradiusLUT.MapControlPointsToLinearSpace()
        effectiveradiusLUT.UseLogScale = 0

    effectiveradiusLUT.NanColor = args.effectiveradius_nan_color

    sd_productspvdDisplay.Opacity = args.sd_products_opacity
    calculator1Display.Opacity = args.calculator1_opacity
    sd_attributespvdDisplay.Opacity = args.sd_attributes_opacity

    renderView1.Update()

apply_presets_logscale_opacity_and_update()


def get_layout():
    layout1 = GetLayout()
    layout1.SetSize(1205, 739)
    renderView1.Update()

get_layout()


def set_current_camera_placement():
    renderView1.InteractionMode = '2D'
    renderView1.CameraPosition = [836.5045867211775, 677.8909274570431, -4098.0762113533165]
    renderView1.CameraFocalPoint = [836.5045867211775, 677.8909274570431, 0.0]
    renderView1.CameraViewUp = [1.0, 0.0, 0.0]
    renderView1.CameraParallelScale = 1060.6601717798205
    renderView1.Update()

set_current_camera_placement()


def axes_settings():
    view = renderView1
    view.ViewSize = [2000, 800]
    view.Background = [1, 1, 1]
    view.CenterAxesVisibility = True
    view.OrientationAxesVisibility = True
    axesGrid = view.AxesGrid
    axesGrid.Visibility = True
    axesGrid.XTitle = 'Z [m]'
    axesGrid.YTitle = 'X [m]'

    axesGrid.XAxisUseCustomLabels = True
    axesGrid.XAxisLabels = [300, 600, 900, 1200]
    axesGrid.YAxisUseCustomLabels = True
    axesGrid.YAxisLabels = [300, 600, 900, 1200]

    axesGrid.XTitleFontSize = 30
    axesGrid.XLabelFontSize = 30
    axesGrid.YTitleFontSize = 30
    axesGrid.YLabelFontSize = 30

    axesGrid.XTitleColor = [0, 0, 0]
    axesGrid.XLabelColor = [0, 0, 0]
    axesGrid.YTitleColor = [0, 0, 0]
    axesGrid.YLabelColor = [0, 0, 0]
    axesGrid.GridColor = [0.1, 0.1, 0.1]
    renderView1.CenterAxesVisibility = False
    renderView1.Update()

axes_settings()


def time_annotation():
    time = AnnotateTimeFilter(guiName = "AnnotateTimeFilter1", Format = 'Time:{time:f}s')
    repr = Show(time, view)
    renderView1.Update()

time_annotation()


# save animation to an Ogg Vorbis file
SaveAnimation('output/anim2.ogv', renderView1, FrameRate=5)
# save animation frame as pdfs
for t in sd_productspvd.TimestepValues:
    renderView1.ViewTime = t
    for reader in (sd_productspvd, sd_attributespvd):
        reader.UpdatePipeline(t)    
        ExportView(
            filename=f'output/anim_frame_{t}.pdf',
            view=renderView1,
            Rasterize3Dgeometry= False,
            GL2PSdepthsortmethod= 'BSP sorting (slow, best)',
     )
RenderAllViews()

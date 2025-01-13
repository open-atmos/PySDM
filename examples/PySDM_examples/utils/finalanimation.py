from paraview.simple import *
import argparse
from pathlib import Path
paraview.simple._DisableFirstRenderCameraReset()


def cli_using_argparse(ap):
    ap.add_argument("product_path", help = 'path to pvd products file')
    ap.add_argument("attributes_path", help=' path to pvd attributes file')
    ap.add_argument('--multiplicity_preset', default = 'Inferno (matplotlib)', help = 'Preset for multiplicity')
    ap.add_argument('--multiplicity_logscale', action = 'store_false', help = 'Use log scale for multiplicity')
    ap.add_argument('--effectiveradius_preset', default = 'Black, Blue and White', help = 'Preset for effectiveradius')
    ap.add_argument('--effectiveradius_logscale', action = 'store_false', help = 'Use log scale for effectiveradius')
    ap.add_argument('--effectiveradius_nan_color', nargs = 3, type = float, default = [0.666, 0.333, 1.0], help = 'Nan color in RGB format for effectiveradius')
    ap.add_argument('--sd_products_opacity', type = float, default = 0.9, help = 'Opacity for sd_products')
    ap.add_argument('--calculator1_opacity', type = float, default = 0.19, help = 'Opacity for calculator1')
    ap.add_argument('--sd_attributes_opacity', type = float, default = 0.77, help = 'Opacity for sd_attributes')

ap = argparse.ArgumentParser()
cli_using_argparse(ap)

args = ap.parse_args()
sd_productspvd = OpenDataFile(args.product_path)
sd_attributespvd = OpenDataFile(args.attributes_path)


#startup_settings
from collections import namedtuple
x = {
    'renderView1': GetActiveViewOrCreate('RenderView')
}
x = namedtuple("X", x.keys())(**x)

multiplicityLUT = GetColorTransferFunction('multiplicity')
sd_attributespvdDisplay = Show(sd_attributespvd, x.renderView1, 'UnstructuredGridRepresentation')
sd_attributespvdDisplay = GetDisplayProperties(sd_attributespvd, view=x.renderView1)
effectiveradiusLUT = GetColorTransferFunction('effectiveradius')
multiplicityLUT.RescaleTransferFunction(19951.0, 50461190157.0)
effectiveradiusLUT.RescaleTransferFunction(0.1380997175392798, 207.7063518856934)
sd_productspvdDisplay = GetDisplayProperties(sd_productspvd, view=x.renderView1)
x.renderView1.Update()


def create_new_calculator(input, representation, function, color_by1, color_by2, color_by3, scalar_coloring = False, hide = False, *, y, registrationame):
    calculator = Calculator(registrationName=registrationame, Input=input)
    display = Show(calculator, y.renderView1, representation)
    calculator.Function = function
    y.renderView1.Update()
    if scalar_coloring == True:
        ColorBy(display, (color_by1, color_by2, color_by3))
    else:
        None
    if hide == True:
        Hide(calculator, y.renderView1)
    else:
        None
    y.renderView1.Update()


def scalar_bar(*, y):
    calculator1Display = Show(calculator1, y.renderView1, 'UnstructuredGridRepresentation')
    calculator1Display.SetScalarBarVisibility(y.renderView1, True)
    scalarBar = GetScalarBar(effectiveradiusLUT, y.renderView1)
    scalarBar.ComponentTitle = ""
    scalarBar.Title = "effective radius [um]"
    y.renderView1.Update()



def create_glyph(registration_name, input, scale_array1, scale_array2, color_by = False, *, y):
    glyph = Glyph(registrationName=registration_name, Input=input,
        GlyphType='Arrow')
    glyphDisplay = Show(glyph, y.renderView1, 'GeometryRepresentation')
    glyphDisplay.Representation = 'Surface'
    glyph.ScaleArray = [scale_array1, scale_array2]
    glyph.ScaleFactor = 100
    glyphDisplay.SetScalarBarVisibility(y.renderView1, True)
    if color_by == True:
        ColorBy(glyphDisplay, None)
    else:
        None
    y.renderView1.Update()



def apply_presets_logscale_opacity_and_update(*, y):
    calculator1Display = Show(calculator1, y.renderView1, 'UnstructuredGridRepresentation')
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

    sd_productspvdDisplay.SetRepresentationType('Surface With Edges')
    sd_productspvdDisplay.Opacity = args.sd_products_opacity
    calculator1Display.Opacity = args.calculator1_opacity
    sd_attributespvdDisplay.Opacity = args.sd_attributes_opacity

    y.renderView1.Update()




def get_layout(*, y):
    layout1 = GetLayout()
    layout1.SetSize(1205, 739)
    y.renderView1.Update()

def set_current_camera_placement(*, y):
    y.renderView1.InteractionMode = '2D'
    y.renderView1.CameraPosition = [836.5045867211775, 677.8909274570431, -4098.0762113533165]
    y.renderView1.CameraFocalPoint = [836.5045867211775, 677.8909274570431, 0.0]
    y.renderView1.CameraViewUp = [1.0, 0.0, 0.0]
    y.renderView1.CameraParallelScale = 1060.6601717798205
    y.renderView1.Update()



def axes_settings(*, view):
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
    view.CenterAxesVisibility = False
    view.Update()




def time_annotation(*, y):
    time = AnnotateTimeFilter(guiName = "AnnotateTimeFilter1", Format = 'Time:{time:f}s')
    repr = Show(time, y.renderView1)
    y.renderView1.Update()



def text(text_in, position_y, *, view): 
    text = Text()
    text.Text = text_in
    textDisplay = Show(text, view)
    textDisplay.Color = [1.0, 1.0, 1.0]  
    textDisplay.WindowLocation = 'Any Location'
    textDisplay.Position = [0.01, position_y]


calculator1 = create_new_calculator(sd_attributespvd, 'UnstructuredGridRepresentation', '"relative fall velocity"*(-iHat)', 'None', 'None', 'None', y=x, registrationame='Calculator1')
scalar_bar(y=x)
glyph1 = create_glyph('Glyph1', calculator1, 'POINTS', 'relative fall velocity', y=x)
calculator2 = create_new_calculator(sd_productspvd, 'StructuredGridRepresentation', 'cx*jHat+cy*iHat', 'CELLS', 'Result', 'Magnitude', True, True, y=x, registrationame='Calculator2')
apply_presets_logscale_opacity_and_update(y=x)
glyph2 = create_glyph('Glyph2', calculator2, 'CELLS', 'Result', True, y=x)
get_layout(y=x)
set_current_camera_placement(y=x)
axes_settings(view=x.renderView1)
time_annotation(y=x)
text("Arrows scale with Courant number C=u*Δt/Δx,", 0.7, view=x.renderView1)
text("reflecting the grid spacing Δx and Δy.", 0.65, view=x.renderView1)

# save animation to an Ogg Vorbis file
SaveAnimation('output/anim2.ogv', x.renderView1, FrameRate=5)
# save animation frame as pdfs
for t in sd_productspvd.TimestepValues:
    x.renderView1.ViewTime = t
    for reader in (sd_productspvd, sd_attributespvd):
        reader.UpdatePipeline(t)    
        ExportView(
            filename=f'output/anim_frame_{t}.pdf',
            view=x.renderView1,
            Rasterize3Dgeometry= False,
            GL2PSdepthsortmethod= 'BSP sorting (slow, best)',
    )
RenderAllViews()

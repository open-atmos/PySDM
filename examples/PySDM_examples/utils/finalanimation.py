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


# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=sd_attributespvd)
calculator1Display = Show(calculator1, renderView1, 'UnstructuredGridRepresentation')
calculator1Display.Representation = 'Surface'
calculator1.Function = '"relative fall velocity"*(-iHat)'
renderView1.Update()


# show and settings color bar legend 
calculator1Display.SetScalarBarVisibility(renderView1, True)
scalarBar = GetScalarBar(effectiveradiusLUT, renderView1)
scalarBar.ComponentTitle = ""
scalarBar.Title = "effective radius [um]"


# create a new 'Glyph'
glyph1 = Glyph(registrationName='Glyph1', Input=calculator1,
    GlyphType='Arrow')
glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')
glyph1Display.Representation = 'Surface'
glyph1.ScaleArray = ['POINTS', 'relative fall velocity']
glyph1Display.SetScalarBarVisibility(renderView1, True)
glyph1.ScaleFactor = 100
renderView1.Update()


# create a new 'Calculator'
calculator2 = Calculator(registrationName='Calculator2', Input=sd_productspvd)
calculator2Display = Show(calculator2, renderView1, 'StructuredGridRepresentation')
calculator2.Function = 'cx*jHat+cy*iHat'
renderView1.Update()
ColorBy(calculator2Display, ('CELLS', 'Result', 'Magnitude'))
Hide(calculator2, renderView1)


# create a new 'Glyph'
glyph2 = Glyph(registrationName='Glyph2', Input=calculator2,
    GlyphType='Arrow')
glyph2Display = Show(glyph2, renderView1, 'GeometryRepresentation')
glyph2Display.Representation = 'Surface'
glyph2.ScaleArray = ['CELLS', 'Result']
glyph2.ScaleFactor = 100
renderView1.Update()
ColorBy(glyph2Display, None)
HideScalarBarIfNotNeeded(effectiveradiusLUT, renderView1)


# apply presets, logscale, opacity, and update.
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


# get layout
layout1 = GetLayout()
layout1.SetSize(1205, 739)


# set current camera placement
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [836.5045867211775, 677.8909274570431, -4098.0762113533165]
renderView1.CameraFocalPoint = [836.5045867211775, 677.8909274570431, 0.0]
renderView1.CameraViewUp = [1.0, 0.0, 0.0]
renderView1.CameraParallelScale = 1060.6601717798205


# axes settings
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


# time annotation
time = AnnotateTimeFilter(guiName = "AnnotateTimeFilter1", Format = 'Time:{time:f}s')
repr = Show(time, view)
renderView1.Update()


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

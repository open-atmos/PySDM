from paraview import simple as pvs
#nowe
import argparse

# load data
#nowe
ap = argparse.ArgumentParser()
ap.add_argument("product_path", help = 'path to pvd products file')
ap.add_argument("attributes_path", help=' path to pvd attributes file')
args = ap.parse_args()
reader_prod = pvs.OpenDataFile(args.product_path)
reader_attr = pvs.OpenDataFile(args.attributes_path)
#reader_prod = pvs.OpenDataFile("./output/sd_products.pvd")
#reader_attr = pvs.OpenDataFile("./output/sd_attributes.pvd")

# prepare view settings
view = pvs.GetRenderView()
view.ViewSize = [2000, 800]
view.Background = [1, 1, 1]
view.CenterAxesVisibility = False
view.OrientationAxesVisibility = False
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

# render particles
var = 'radius'
multiplier = 1e6
palette = 'Cold and Hot'
palette_invert = False
color_range = [0, 10]
logscale = False
title = var + ' [um]'

calculator = pvs.Calculator(reader_attr)
calculator.Function = f'{var}*{multiplier}'
display_attr = pvs.Show(calculator, view)

display_attr.SetRepresentationType('Point Gaussian')
display_attr.ShaderPreset = 'Sphere'
display_attr.GaussianRadius = 5
display_attr.MapScalars = 1

display_attr.Ambient = .25
pvs.ColorBy(display_attr, ('POINTS', 'Result'))
color_scale_attr = pvs.GetColorTransferFunction('Result')
color_scale_attr.ApplyPreset(palette, True)
if palette_invert:
    color_scale_attr.InvertTransferFunction()
if color_range is None:
    display_attr.RescaleTransferFunctionToDataRange(True)
else:
    color_scale_attr.RescaleTransferFunction(color_range)
if logscale:
    color_scale_attr.MapControlPointsToLogSpace()
    color_scale_attr.UseLogScale = 1
colorbar_attr = pvs.GetScalarBar(color_scale_attr, view)
colorbar_attr.TitleColor = [0, 0, 0]
colorbar_attr.LabelColor = [0, 0, 0]
colorbar_attr.Title = title
colorbar_attr.ComponentTitle = ''
colorbar_attr.TitleFontSize = 30
colorbar_attr.LabelFontSize = 30
colorbar_attr.Visibility = True
colorbar_attr.WindowLocation = 'Any Location'
colorbar_attr.Position = [.1, .333]
colorbar_attr.RangeLabelFormat = '%g'

# render product
var = 'effective radius'
palette = 'X Ray'
palette_invert = True
color_range = [0 , 10]
logscale = False
title = var + ' [um]'

display_prod = pvs.Show(reader_prod)
display_prod.SetRepresentationType('Surface')
display_prod.Ambient = .25
pvs.ColorBy(display_prod, ('CELLS', var))
color_scale_prod = pvs.GetColorTransferFunction(var)
if color_range is None:
    display_prod.RescaleTransferFunctionToDataRange(True)
else:
    color_scale_prod.RescaleTransferFunction(color_range)
color_scale_prod.ApplyPreset(palette, True)
#nowe
color_scale_prod.NanColor = [0.0, 0.0, 0.0]
if palette_invert:
    color_scale_prod.InvertTransferFunction()
colorbar_prod = pvs.GetScalarBar(color_scale_prod, view)
colorbar_prod.TitleColor = [0, 0, 0]
colorbar_prod.LabelColor = [0, 0, 0]
colorbar_prod.Title = title
colorbar_prod.ComponentTitle = ''
colorbar_prod.TitleFontSize = 30
colorbar_prod.LabelFontSize = 30
colorbar_prod.Visibility = True
colorbar_prod.Position = [.92, .333]
colorbar_prod.WindowLocation = 'Any Location'
colorbar_prod.RangeLabelFormat = '%g'

# COS nowe, particles i product in the same time in the same picture, do not cover each other
render_view = pvs.GetActiveViewOrCreate('RenderView')
prod_display = pvs.Show(reader_prod, render_view)
product_display = pvs.Show(reader_prod, render_view)
attr_display = pvs.Show(reader_attr, render_view)
prod_display.Representation = 'Surface'
attr_display.Representation = 'Surface'
prod_display.Opacity = 0.3

# compose the scene
scene = pvs.GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
pvs.Render(view)
cam = pvs.GetActiveCamera()
cam.SetViewUp(1, 0, 0)
pos = list(cam.GetPosition())
pos[-1] = -pos[-1]
cam.SetPosition(pos)
cam.Dolly(1.45)

# save animation to an Ogg Vorbis file
pvs.SaveAnimation('output/anim.ogv', view, FrameRate=10)


# save animation frame as pdfs
for t in reader_prod.TimestepValues:
    view.ViewTime = t
    for reader in (reader_prod, reader_attr):
        reader.UpdatePipeline(t)
    pvs.ExportView(
        filename=f'output/anim_frame_{t}.pdf',
        view=view,
        Rasterize3Dgeometry= False,
        GL2PSdepthsortmethod= 'BSP sorting (slow, best)',
    )
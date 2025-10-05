# to run pvpython script use command line: pvpython filename.py
from paraview import simple as pvs  # Updated import statement

#### disable automatic camera reset on 'Show'
pvs._DisableFirstRenderCameraReset()

# create a new 'PVD Reader'
sd_productspvd = pvs.PVDReader(
    registrationName="sd_products.pvd",
    FileName="C:\\Users\\strza\\Desktop\\PySDM\\examples\\PySDM_examples\\Pyrcel\\output\\sd_products.pvd",
)

# get animation scene
animationScene1 = pvs.GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = pvs.GetActiveViewOrCreate("RenderView")

# show data in view
sd_productspvdDisplay = pvs.Show(
    sd_productspvd, renderView1, "UnstructuredGridRepresentation"
)

# trace defaults for the display properties.
sd_productspvdDisplay.Representation = "Surface"

# reset view to fit data
renderView1.ResetCamera(False, 0.9)

# show color bar/color legend
sd_productspvdDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'RH'
rHLUT = pvs.GetColorTransferFunction("RH")

# get opacity transfer function/opacity map for 'RH'
rHPWF = pvs.GetOpacityTransferFunction("RH")

# get 2D transfer function for 'RH'
rHTF2D = pvs.GetTransferFunction2D("RH")

# create a new 'PVD Reader'
sd_attributespvd = pvs.PVDReader(
    registrationName="sd_attributes.pvd",
    FileName="C:\\Users\\strza\\Desktop\\PySDM\\examples\\PySDM_examples\\Pyrcel\\output\\sd_attributes.pvd",
)

# show data in view
sd_attributespvdDisplay = pvs.Show(
    sd_attributespvd, renderView1, "UnstructuredGridRepresentation"
)

# trace defaults for the display properties.
sd_attributespvdDisplay.Representation = "Surface"

# show color bar/color legend
sd_attributespvdDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'volume'
volumeLUT = pvs.GetColorTransferFunction("volume")

# get opacity transfer function/opacity map for 'volume'
volumePWF = pvs.GetOpacityTransferFunction("volume")

# get 2D transfer function for 'volume'
volumeTF2D = pvs.GetTransferFunction2D("volume")

# get color legend/bar for rHLUT in view renderView1
rHLUTColorBar = pvs.GetScalarBar(rHLUT, renderView1)

# Properties modified on rHLUTColorBar
rHLUTColorBar.LabelColor = [0.0, 0.0, 0.16000610360875867]
rHLUTColorBar.DrawScalarBarOutline = 1
rHLUTColorBar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
rHLUTColorBar.TitleColor = [0.0, 0.0, 0.0]

# Rescale transfer function
rHLUT.RescaleTransferFunction(90.0, 101.0)

# Rescale transfer function
rHPWF.RescaleTransferFunction(90.0, 101.0)

# Rescale 2D transfer function
rHTF2D.RescaleTransferFunction(90.0, 101.0, 0.0, 1.0)

# Properties modified on sd_productspvdDisplay
sd_productspvdDisplay.Opacity = 0.4

# Properties modified on sd_productspvdDisplay
sd_productspvdDisplay.DisableLighting = 1

# Properties modified on sd_productspvdDisplay
sd_productspvdDisplay.Diffuse = 0.76

# Properties modified on sd_productspvdDisplay.DataAxesGrid
sd_productspvdDisplay.DataAxesGrid.GridAxesVisibility = 1

# Properties modified on sd_productspvdDisplay.DataAxesGrid
sd_productspvdDisplay.DataAxesGrid.XTitle = ""
sd_productspvdDisplay.DataAxesGrid.YTitle = ""
sd_productspvdDisplay.DataAxesGrid.ZTitle = ""
sd_productspvdDisplay.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
sd_productspvdDisplay.DataAxesGrid.ShowGrid = 1
sd_productspvdDisplay.DataAxesGrid.LabelUniqueEdgesOnly = 0
sd_productspvdDisplay.DataAxesGrid.XAxisUseCustomLabels = 1
sd_productspvdDisplay.DataAxesGrid.YAxisUseCustomLabels = 1
sd_productspvdDisplay.DataAxesGrid.ZAxisUseCustomLabels = 1

# Properties modified on renderView1
renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.16000610360875867]

# Properties modified on renderView1
renderView1.OrientationAxesOutlineColor = [0.0, 0.0, 0.16000610360875867]

# Properties modified on renderView1
renderView1.OrientationAxesXVisibility = 0

# Properties modified on renderView1
renderView1.OrientationAxesYVisibility = 0

# Properties modified on renderView1
renderView1.OrientationAxesZColor = [0.0, 0.0, 0.0]

# Properties modified on renderView1
renderView1.UseColorPaletteForBackground = 0

# Properties modified on renderView1
renderView1.Background = [1.0, 1.0, 1.0]

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
rHLUT.ApplyPreset("Black, Blue and White", True)

# Properties modified on rHLUT
rHLUT.NanColor = [0.6666666666666666, 1.0, 1.0]

# change scalar bar placement
rHLUTColorBar.Position = [0.9163059163059163, 0.010638297872340425]
# get color legend/bar for volumeLUT in view renderView1
volumeLUTColorBar = pvs.GetScalarBar(volumeLUT, renderView1)

# Properties modified on volumeLUTColorBar
volumeLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
volumeLUTColorBar.DrawScalarBarOutline = 1
volumeLUTColorBar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
volumeLUTColorBar.TitleColor = [0.0, 0.0, 0.0]

# Rescale transfer function
volumeLUT.RescaleTransferFunction(1e-18, 1e-13)

# Rescale transfer function
volumePWF.RescaleTransferFunction(1e-18, 1e-13)

# Rescale 2D transfer function
volumeTF2D.RescaleTransferFunction(1e-18, 1e-13, 0.0, 1.0)

# Properties modified on sd_attributespvdDisplay
sd_attributespvdDisplay.PointSize = 13.0

# Properties modified on sd_attributespvdDisplay
sd_attributespvdDisplay.RenderPointsAsSpheres = 1

# Properties modified on sd_attributespvdDisplay
sd_attributespvdDisplay.Interpolation = "PBR"

# Properties modified on sd_attributespvdDisplay.DataAxesGrid
sd_attributespvdDisplay.DataAxesGrid.GridAxesVisibility = 1

# Properties modified on sd_attributespvdDisplay.DataAxesGrid
sd_attributespvdDisplay.DataAxesGrid.XTitle = ""
sd_attributespvdDisplay.DataAxesGrid.YTitle = ""
sd_attributespvdDisplay.DataAxesGrid.ZTitle = "Z [m]"
sd_attributespvdDisplay.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
sd_attributespvdDisplay.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
sd_attributespvdDisplay.DataAxesGrid.XAxisUseCustomLabels = 1
sd_attributespvdDisplay.DataAxesGrid.YAxisUseCustomLabels = 1

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
volumeLUT.ApplyPreset("Cold and Hot", True)

# convert to log space
volumeLUT.MapControlPointsToLogSpace()

# Properties modified on volumeLUT
volumeLUT.UseLogScale = 1

# Properties modified on volumeLUT
volumeLUT.NumberOfTableValues = 16

# invert the transfer function
volumeLUT.InvertTransferFunction()

# change scalar bar placement
volumeLUTColorBar.Position = [0.9199134199134199, 0.6586879432624113]

layout1 = pvs.GetLayout()
layout1.SetSize(1593, 1128)
# current camera placement for renderView1
renderView1.CameraPosition = [1548.945972263949, -1349.493616194682, 699.2699178747185]
renderView1.CameraFocalPoint = [
    -1.3686701146742275e-13,
    3.1755031232028544e-13,
    505.00000000000017,
]
renderView1.CameraViewUp = [
    -0.07221292769632315,
    0.06043215330151439,
    0.9955567527373153,
]
renderView1.CameraParallelScale = 534.07781536883

output_animation_path = "C:\\Users\\strza\\Desktop\\PySDM\\examples\\PySDM_examples\\Pyrcel\\output_animation.mp4"  # Change the extension as needed
output_screenshot_path = (
    "C:\\Users\\strza\\Desktop\\PySDM\\examples\\PySDM_examples\\Pyrcel\\last_frame.png"
)
pvs.SaveAnimation(
    output_animation_path, renderView1, FrameRate=15
)  # Adjust FrameRate as needed
pvs.SaveScreenshot(output_screenshot_path, renderView1)

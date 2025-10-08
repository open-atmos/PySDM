import argparse
from paraview import simple as pvs  # Updated import statement


def cli_using_argparse(ap):
    ap.add_argument(
        "--sd-products-pvd",
        dest="sd_products_pvd",
        default=r"C:\\Users\\strza\\Desktop\\PySDM\\examples\\PySDM_examples\\Pyrcel\\output\\sd_products.pvd",
        help="Path to sd_products.pvd",
    )
    ap.add_argument(
        "--sd-attributes-pvd",
        dest="sd_attributes_pvd",
        default=r"C:\\Users\\strza\\Desktop\\PySDM\\examples\\PySDM_examples\\Pyrcel\\output\\sd_attributes.pvd",
        help="Path to sd_attributes.pvd",
    )


parser = argparse.ArgumentParser(
    description="ParaView pvpython script for visualizing sd_products and sd_attributes PVD files."
)
cli_using_argparse(parser)
args = parser.parse_args()

pvs._DisableFirstRenderCameraReset()

# get active view
renderView1 = pvs.GetActiveViewOrCreate("RenderView")

# show data in view prod
sd_productspvd = pvs.PVDReader(
    registrationName="sd_products.pvd",
    FileName=args.sd_products_pvd,
)
sd_productspvdDisplay = pvs.Show(
    sd_productspvd, renderView1, "UnstructuredGridRepresentation"
)
sd_productspvdDisplay.Representation = "Surface"
sd_productspvdDisplay.SetScalarBarVisibility(renderView1, True)
rHLUT = pvs.GetColorTransferFunction("RH")
rHLUT.RescaleTransferFunction(90.0, 101.0)

# show data in view attr
sd_attributespvd = pvs.PVDReader(
    registrationName="sd_attributes.pvd",
    FileName=args.sd_attributes_pvd,
)
sd_attributespvdDisplay = pvs.Show(
    sd_attributespvd, renderView1, "UnstructuredGridRepresentation"
)
sd_attributespvdDisplay.Representation = "Surface"
sd_attributespvdDisplay.SetScalarBarVisibility(renderView1, True)
volumeLUT = pvs.GetColorTransferFunction("volume")
volumeLUT.RescaleTransferFunction(1e-18, 1e-13)

# Properties modified on rHLUTColorBar
rHLUTColorBar = pvs.GetScalarBar(rHLUT, renderView1)
rHLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
rHLUTColorBar.DrawScalarBarOutline = 1
rHLUTColorBar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
rHLUTColorBar.TitleColor = [0.0, 0.0, 0.0]

# lightning and opacity sd_productspvdDisplay
sd_productspvdDisplay.Opacity = 0.4
sd_productspvdDisplay.DisableLighting = 1
sd_productspvdDisplay.Diffuse = 0.76
rHLUT.ApplyPreset("Black, Blue and White", True)
rHLUT.NanColor = [0.6666666666666666, 1.0, 1.0]

# Properties modified on sd_productspvdDisplay.DataAxesGrid
sd_productspvdDisplay.DataAxesGrid.GridAxesVisibility = 1
sd_productspvdDisplay.DataAxesGrid.XTitle = ""
sd_productspvdDisplay.DataAxesGrid.YTitle = ""
sd_productspvdDisplay.DataAxesGrid.ZTitle = ""
sd_productspvdDisplay.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
sd_productspvdDisplay.DataAxesGrid.ShowGrid = 1
sd_productspvdDisplay.DataAxesGrid.LabelUniqueEdgesOnly = 0
sd_productspvdDisplay.DataAxesGrid.XAxisUseCustomLabels = 1
sd_productspvdDisplay.DataAxesGrid.YAxisUseCustomLabels = 1
sd_productspvdDisplay.DataAxesGrid.ZAxisUseCustomLabels = 1

# orientation axes
renderView1.OrientationAxesLabelColor = [0.0, 0.0, 0.0]
renderView1.OrientationAxesOutlineColor = [0.0, 0.0, 0.0]
renderView1.OrientationAxesXVisibility = 0
renderView1.OrientationAxesYVisibility = 0
renderView1.OrientationAxesZColor = [0.0, 0.0, 0.0]

# background
renderView1.UseColorPaletteForBackground = 0
renderView1.Background = [1.0, 1.0, 1.0]

# Properties modified on volumeLUTColorBar
volumeLUTColorBar = pvs.GetScalarBar(volumeLUT, renderView1)
volumeLUTColorBar.LabelColor = [0.0, 0.0, 0.0]
volumeLUTColorBar.DrawScalarBarOutline = 1
volumeLUTColorBar.ScalarBarOutlineColor = [0.0, 0.0, 0.0]
volumeLUTColorBar.TitleColor = [0.0, 0.0, 0.0]

# attr size and shape and color etc
sd_attributespvdDisplay.PointSize = 13.0
sd_attributespvdDisplay.RenderPointsAsSpheres = 1
sd_attributespvdDisplay.Interpolation = "PBR"
volumeLUT.ApplyPreset("Cold and Hot", True)
volumeLUT.MapControlPointsToLogSpace()
volumeLUT.UseLogScale = 1
volumeLUT.NumberOfTableValues = 16
volumeLUT.InvertTransferFunction()

# Properties modified on sd_attributespvdDisplay.DataAxesGrid
sd_attributespvdDisplay.DataAxesGrid.GridAxesVisibility = 1
sd_attributespvdDisplay.DataAxesGrid.XTitle = ""
sd_attributespvdDisplay.DataAxesGrid.YTitle = ""
sd_attributespvdDisplay.DataAxesGrid.ZTitle = "Z [m]"
sd_attributespvdDisplay.DataAxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
sd_attributespvdDisplay.DataAxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
sd_attributespvdDisplay.DataAxesGrid.XAxisUseCustomLabels = 1
sd_attributespvdDisplay.DataAxesGrid.YAxisUseCustomLabels = 1

# current camera placement for renderView1 and layout
layout1 = pvs.GetLayout()
layout1.SetSize(1592, 1128)
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

# get animation scene
animationScene1 = pvs.GetAnimationScene()
animationScene1.UpdateAnimationUsingDataTimeSteps()

output_animation_path = r"C:\Users\strza\Desktop\PySDM\examples\PySDM_examples\Pyrcel\output_animation.avi"  # Change the extension as needed
output_screenshot_path = (
    r"C:\Users\strza\Desktop\PySDM\examples\PySDM_examples\Pyrcel\last_frame.png"
)
pvs.SaveAnimation(output_animation_path, renderView1, FrameRate=15)
pvs.SaveScreenshot(output_screenshot_path, renderView1)

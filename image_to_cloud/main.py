#!/usr/bin/env python3
# shebang line for linux / mac

from copy import deepcopy
from functools import partial
import glob
from random import randint
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import argparse
import open3d as o3d
import os


#-----------------------------
# Layout Configurations
#-----------------------------
def onLayout(window, controlPanel, viewport, ctx):

    # Rectangle size
    rectangleSize = window.content_rect

    # Layout and scene frame
    controlPanel.frame = o3d.visualization.gui.Rect(rectangleSize.width - 210, 20, 210, rectangleSize.height - 80)
    viewport.frame = o3d.visualization.gui.Rect(rectangleSize.x, rectangleSize.y, rectangleSize.width, rectangleSize.height)

#-----------------------------
# Viewport Configuration
#-----------------------------
def viewportConfiguration(window, pointClouds):
    viewport = o3d.visualization.gui.SceneWidget()
    viewport.scene = o3d.visualization.rendering.Open3DScene(window.renderer)

    # Material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    for idx, pointCloud in enumerate(pointClouds):
        viewport.scene.add_geometry(f"cloud_{idx}", pointCloud, material)
        viewport.scene.show_geometry(f"cloud_{idx}", False)  # Hide 

    return viewport

#-----------------------------
# Checkbox Configurations
#-----------------------------
def onCheckboxShowOriginalToggled(isChecked, viewport, idx):

    cloudName = f"cloud_{idx}"

    if isChecked:
        viewport.scene.show_geometry(cloudName, True)   # Show Again  
    else:
        viewport.scene.show_geometry(cloudName, False)  # Hide  

#-----------------------------
# controlPanel Configuration
#-----------------------------
def controlPanelConfiguration(window, pointClouds, viewport):

    #Layout Configuration
    controlPanel = o3d.visualization.gui.Vert(0.25 * window.theme.font_size,
        o3d.visualization.gui.Margins(10, 10, 10, 10))

    # Create and activate checkboxes/label 
    for idx, _ in enumerate(pointClouds):

        #Create a label for each pcd and add the color name
        label = o3d.visualization.gui.Label(f"Point Cloud {idx}")

        checkboxShowOriginal = o3d.visualization.gui.Checkbox(f"Show PointCloud")
        colorPicker = o3d.visualization.gui.ColorEdit()

        checkboxShowOriginal.set_on_checked(partial(onCheckboxShowOriginalToggled, viewport, idx))
        #colorPicker.set_on_value_changed(partial(onColorChange, idx = idx))

        controlPanel.add_child(label)
        controlPanel.add_child(checkboxShowOriginal)
        controlPanel.add_child(colorPicker)

    return controlPanel

view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [10.0, 34.024543762207031, 11.225864410400391],
                "boundingbox_min": [-39.714397430419922, -16.512752532958984, -1.9472264051437378],
                "field_of_view": 60.0,
                "front": [0.87911045824568079, -0.1143707949631662, 0.46269225567601935],
                "lookat": [-14.857198715209961, 8.7558956146240234, 4.6393190026283264],
                "up": [-0.45122740480118839, 0.11291073802962912, 0.88523725316662361],
                "zoom": 0.53999999999999981
            }
        ],
    "version_major": 1,
    "version_minor": 0
}


def main():

    # -----------------------------
    #  Parse arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Read Tum dataset rgb and depth")
    parser.add_argument('-ifd', "--inputFolderDepth",  default='tum_dataset/depth', type=str,
        help="Folder where the depth images are.)")
    parser.add_argument('-ifrgb', "--inputFolderRGB",  default='tum_dataset/rgb', type=str,
        help="Folder where the rgb images are.)")
    parser.add_argument('-outpcd', "--outputPointClouds", default='cloud', type=str,
        help="Folder where the rgb images are.)")
    
    args = vars(parser.parse_args())

    # ----------------------------------
    # Search for images (rgb and depth) in input folder
    # ----------------------------------
    imageFilenamesDepth = glob.glob(os.path.join(args['inputFolderDepth'], '*.png'))
    imageFilenamesDepth.sort() # To make sure we have alphabetic order
    print('Found input point clouds RGB: ' + str(imageFilenamesDepth))
    
    imageFilenamesRGB = glob.glob(os.path.join(args['inputFolderRGB'], '*.png'))
    imageFilenamesRGB.sort() # To make sure we have alphabetic order
    print('Found input point clouds RGB: ' + str(imageFilenamesRGB))
    
    if len(imageFilenamesDepth) == 0:
        raise ValueError('Could not find any depth in folder ' + args['inputFolderDepth'])
    
    if len(imageFilenamesRGB) == 0:
        raise ValueError('Could not find any rgb in folder ' + args['inputFolderRGB'])
    
    # ----------------------------------
    # Transform in RGBD
    # ----------------------------------
    rgbdList = []
    for filenameDepth, filenameRGB in zip(imageFilenamesDepth, imageFilenamesRGB):

        if not os.path.exists(filenameDepth):
            raise ValueError(f"File not found: {filenameDepth}")
        if not os.path.exists(filenameRGB):
            raise ValueError(f"File not found: {filenameRGB}")

        print(f"Loaded {filenameDepth}")
        print(f"Loaded {filenameRGB}")

        # Read the images as Open3D Image objects
        depthOpen3d = o3d.io.read_image(filenameDepth)
        rgbOpen3d = o3d.io.read_image(filenameRGB)

        # Create RGBD image from color + depth
        rgbdImage = o3d.geometry.RGBDImage.create_from_tum_format(rgbOpen3d, depthOpen3d)

        rgbdList.append(rgbdImage)

    # ----------------------------------
    # Transform in PointClouds
    # ----------------------------------
    pointClouds = []
    for rgbd in rgbdList:
        pointClouds.append(o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)))
        
    # ------------------------------------
    # Write the point cloud
    # ------------------------------------   
    for i, pcd in enumerate(pointClouds):
        filename = os.path.join(args['outputPointClouds'], f"cloud_{i:02d}.ply")
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Saved: {filename}")

    # ------------------------------------
    # Visualize the point cloud
    # ------------------------------------   

    axes_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)

    # App inicialization
    application = o3d.visualization.gui.Application.instance
    application.initialize()

    # Create window
    window = application.create_window("GUI", 1280, 800)

    viewport = viewportConfiguration(window, pointClouds)

    controlPanel = controlPanelConfiguration(window, pointClouds, viewport)


    # Update new Scene and Layout
    window.add_child(viewport)
    window.add_child(controlPanel)

    # Handles viewport + control panel positioning
    window.set_on_layout(partial(onLayout, window, controlPanel, viewport))

    # Run the app
    application.run()

if __name__ == '__main__':
    main()
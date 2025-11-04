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
import copy
from open3d.geometry import KDTreeSearchParamHybrid
from open3d.pipelines.registration import compute_fpfh_feature
from open3d.pipelines.registration import registration_ransac_based_on_feature_matching



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
# Paint Point Clouds Configurations
#-----------------------------
def onColorChange(viewport, newColor, idx):

    # For transformed cloud
    cloudName = f"cloud_{idx}"

    # Create new color for the material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = (newColor.red, newColor.green, newColor.blue, 1.0)

    # Update material in viewport
    viewport.scene.modify_geometry_material(cloudName, material)

def onColorChangeRegistartionDebug(viewport, newColor, idx):

    # For transformed cloud
    cloudName = f"cloud_registered_{idx}"

    # Create new color for the material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = (newColor.red, newColor.green, newColor.blue, 1.0)

    # Update material in viewport
    viewport.scene.modify_geometry_material(cloudName, material)

#-----------------------------
# Viewport Configuration
#-----------------------------
def viewportConfiguration(window, pointClouds, pointCloudsRegistrationDebug):
    viewport = o3d.visualization.gui.SceneWidget()
    viewport.scene = o3d.visualization.rendering.Open3DScene(window.renderer)

    # Material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    for idx, pointCloud in enumerate(pointClouds):
        viewport.scene.add_geometry(f"cloud_{idx}", pointCloud, material)
        viewport.scene.show_geometry(f"cloud_{idx}", False)  # Hide 
    
    for idx, pointCloud in enumerate(pointCloudsRegistrationDebug):
        viewport.scene.add_geometry(f"cloud_registered_{idx}", pointCloud, material)
        viewport.scene.show_geometry(f"cloud_registered_{idx}", False)  # Hide 

    

    # Setup camera to start with a good viewpoint 
    if pointClouds:
        bounds = pointClouds[0].get_axis_aligned_bounding_box()
        for pointCloud in pointClouds[1:]:
                bounds += pointCloud.get_axis_aligned_bounding_box()
                viewport.setup_camera(60, bounds, bounds.get_center())

        viewport.background_color = o3d.visualization.gui.Color(1, 1, 1)

    return viewport

#-----------------------------
# Checkbox Configurations
#-----------------------------
def onCheckboxShowOriginalToggled(viewport, idx, isChecked):

    cloudName = f"cloud_{idx}"

    if isChecked:
        viewport.scene.show_geometry(cloudName, True)   # Show Again  
    else:
        viewport.scene.show_geometry(cloudName, False)  # Hide 

def onCheckboxShowRegistrationDebugToggled(viewport, idx, isChecked):

    cloudName = f"cloud_registered_{idx}"

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

        checkboxShowOriginal = o3d.visualization.gui.Checkbox(f"Show PointCloud Original")
        checkboxShowRegistrationDebug = o3d.visualization.gui.Checkbox(f"Show PointCloud Registered")
        colorPickerOriginal = o3d.visualization.gui.ColorEdit()
        colorPickerRegistrationDebug = o3d.visualization.gui.ColorEdit()

        checkboxShowOriginal.set_on_checked(partial(onCheckboxShowOriginalToggled, viewport, idx))
        checkboxShowRegistrationDebug.set_on_checked(partial(onCheckboxShowRegistrationDebugToggled, viewport, idx))
        colorPickerOriginal.set_on_value_changed(partial(onColorChange, viewport, idx = idx))
        colorPickerRegistrationDebug.set_on_value_changed(partial(onColorChangeRegistartionDebug, viewport, idx = idx))

        controlPanel.add_child(label)
        controlPanel.add_child(checkboxShowOriginal)
        controlPanel.add_child(colorPickerOriginal)
        controlPanel.add_child(checkboxShowRegistrationDebug)
        controlPanel.add_child(colorPickerRegistrationDebug)

    return controlPanel

#--------------------------
# Downsample and Normals
#-------------------------
def downsampleAndEstimateNormals(pointCloud, args):
                radiusNormal = args['voxelSize'] * 2
                pointCloudDownsampled = pointCloud.voxel_down_sample(args['voxelSize'])
                pointCloudDownsampled.estimate_normals(KDTreeSearchParamHybrid(radius=radiusNormal, max_nn=30)) # type: ignore

                return pointCloudDownsampled


#--------------------------
# Global Regist PCs
#--------------------------
def calculateGlobalRegistrationTransformation(accumulatedPointCloudDownsampled, pointCloudDownsampled, args):

    ransacDistanceThreshold = args['voxelSize'] * 1.5
    radiusFeature = args['voxelSize'] * 5

    #Compute features
    pointCloudFeatures = compute_fpfh_feature(pointCloudDownsampled,
        KDTreeSearchParamHybrid(radius=radiusFeature, max_nn=100))
    accumulatedPointCloudFeatures = compute_fpfh_feature(accumulatedPointCloudDownsampled,
        KDTreeSearchParamHybrid(radius=radiusFeature, max_nn=100))

    #Compute transformation
    globalRegistrationTransformation = registration_ransac_based_on_feature_matching(
            pointCloudDownsampled, accumulatedPointCloudDownsampled, 
            pointCloudFeatures, accumulatedPointCloudFeatures, 
            True, ransacDistanceThreshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,  # number of RANSAC iterations per sample
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransacDistanceThreshold)
            ], 
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)) # iteration limit and confidence

    return globalRegistrationTransformation


#--------------------------
# ICP PCs
#--------------------------
def calculateICPTransformation(pointCloud, accumulatedPointCloud, globalRegistrationTransformation, args):
    distanceThreshold = args['voxelSize'] * 0.5
    radiusNormal = args['voxelSize'] * 2

    # Compute normals
    pointCloud.estimate_normals(KDTreeSearchParamHybrid(radius=radiusNormal, max_nn=30)) # type: ignore
    accumulatedPointCloud.estimate_normals(KDTreeSearchParamHybrid(radius=radiusNormal, max_nn=30)) # type: ignore

    # run icp
    icpRegistrationTransformation = o3d.pipelines.registration.registration_icp(
            pointCloud, accumulatedPointCloud,
            distanceThreshold,
            globalRegistrationTransformation.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    return icpRegistrationTransformation

def main():

    # -----------------------------
    #  Parse arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Read Poit Clouds To Register")
    parser.add_argument('-ifd', "--inputFolder",  default='../image_to_cloud/cloud', type=str,
        help="Folder where the pointclouds images are.)")
    parser.add_argument('-outpcd', "--outputPointClouds", default='cloud_registered', type=str,
        help="Folder where the rgb images are.)")
    parser.add_argument('-voxS', "--voxelSize", default= 0.05, type=float,
        help="Voxel Size.)")
    
    args = vars(parser.parse_args())

    # ----------------------------------
    # Search for points clouds in input folder
    # ----------------------------------
    pointCloudFilenames = glob.glob(os.path.join(args['inputFolder'], '*.pcd'))
    pointCloudFilenames.sort() # To make sure we have alphabetic order
    print('Found input point clouds: ' + str(pointCloudFilenames))
    if len(pointCloudFilenames) == 0:
        raise ValueError('Could not find any point cloud in folder ' + args['inputFolder'])
    
    # ----------------------------------
    # Load PCs
    # ----------------------------------
    print("Loading point clouds ...")
    pointClouds = []
    for filename in pointCloudFilenames:
        if not os.path.exists(filename):
            raise ValueError(f"File not found: {filename}")

        print('Loaded ' f"{filename}")
        pointClouds.append(o3d.io.read_point_cloud(filename))

    
    #-----------------------------
    # Registration procedure
    #-----------------------------
    registeredTable = {} #only for debug table

    pointCloudsRegistrationDebug = copy.deepcopy(pointClouds)
    accumulatedPointCloud = o3d.geometry.PointCloud()
    accumulatedPointCloudDownsampled = o3d.geometry.PointCloud()

    for idx, pointCloud in enumerate(pointCloudsRegistrationDebug):
        
        #Create a list with range idx
        numberPC = list(range(idx))

        #Create a string with - (0-1-2...)
        listText = " - ".join(str(number) for number in numberPC)
        
        if len(accumulatedPointCloud.points) == 0: # no points in the accumulated

            estimatedTransformation = np.eye(4) # Transformation is identity for the first cloud

            pointCloudDownsampled = downsampleAndEstimateNormals(pointCloud, args)

        else:
            # Downsample and Estimate Normals
            pointCloudDownsampled = downsampleAndEstimateNormals(pointCloud, args)

            # Global Registration procedure 
            globalRegistrationTransformation = calculateGlobalRegistrationTransformation(accumulatedPointCloudDownsampled, pointCloudDownsampled,  args)
            print(f"[RANSAC] [{listText}] -- {idx} done")
            
            # ICP Registration procedure 
            icpRegistrationTransformation = calculateICPTransformation(pointCloud, accumulatedPointCloud, globalRegistrationTransformation, args)
            print(f"[ICP] [{listText}] -- {idx} done")

            estimatedTransformation = icpRegistrationTransformation.transformation

            
        # -----------------------------------------
        # Apply transformation and accumulate
        # -----------------------------------------
        # Apply transformation 
        accumulatedPointCloud += pointCloud.transform(estimatedTransformation)

        # Accumulate transformed Clouds                                  
        accumulatedPointCloudDownsampled += pointCloudDownsampled.transform(estimatedTransformation) # type: ignore
        # Post merge downlsampling
        accumulatedPointCloudDownsampled = accumulatedPointCloudDownsampled.voxel_down_sample(args['voxelSize'])
        
        # -----------------------------------------
        # Update dictionary registeredTable and print Accumulated PC Points
        # -----------------------------------------
        if numberPC == []:
            pass
        else:
            #Create a List with fitness and inlier rme
            registeredValues = [
                f"{globalRegistrationTransformation.fitness:.4f}",
                f"{globalRegistrationTransformation.inlier_rmse:.4f}",
                f"{icpRegistrationTransformation.fitness:.4f}",
                f"{icpRegistrationTransformation.inlier_rmse:.4f}"
            ]
            
            # Save the values in a dictionary
            registeredTable[f"[{listText}] -- {idx}"] = registeredValues

            # Print accumulated points
            print(f'Points in the accumulated cloud [{listText}] -- {idx}: ' + str(len(accumulatedPointCloud.points)))


    # -----------------------------------------
    # Table for debug print
    # -----------------------------------------
    print(f"{'Point Clouds':<25} | {'RANSAC Fit':<16} | {'RANSAC RMSE':<16} | {'ICP Fit':<16} | {'ICP_RMSE':<16}")
    print("-" * 93)
    for key, values in registeredTable.items():
        print(f"{key:<25} | {values[0]:<16} | {values[1]:<16} | {values[2]:<16} | {values[3]:<16}")

    # ------------------------------------
    # Write the point cloud
    # ------------------------------------   
    filename = os.path.join(args['outputPointClouds'], f"cloud_registered.pcd")
    o3d.io.write_point_cloud(filename, accumulatedPointCloudDownsampled)
    print(f"Saved: {filename}")

    # ------------------------------------
    # Visualize the point cloud
    # ------------------------------------   

    # App inicialization
    application = o3d.visualization.gui.Application.instance
    application.initialize()

    # Create window
    window = application.create_window("GUI", 1280, 800)

    viewport = viewportConfiguration(window, pointClouds, pointCloudsRegistrationDebug)

    controlPanel = controlPanelConfiguration(window, pointClouds, viewport)

    # Update new Scene and Layout
    window.add_child(viewport)
    window.add_child(controlPanel)

    # # Handles viewport + control panel positioning
    window.set_on_layout(partial(onLayout, window, controlPanel, viewport))

    # Run the app
    application.run()

if __name__ == '__main__':
    main()
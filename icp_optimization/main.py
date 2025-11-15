#!/usr/bin/env python3
from copy import deepcopy
from functools import partial
import glob
from random import randint
import numpy as np
import argparse
import open3d as o3d
import os
import copy
from open3d.geometry import KDTreeSearchParamHybrid
from open3d.pipelines.registration import compute_fpfh_feature
from open3d.pipelines.registration import registration_ransac_based_on_feature_matching
from custom_icp import CustomICP
from scipy.optimize import least_squares



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
    # For original cloud
    cloudName = f"cloud_{idx}"

    # Create new color for the material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = (newColor.red, newColor.green, newColor.blue, 1.0)

    # Update material in viewport
    viewport.scene.modify_geometry_material(cloudName, material)

def onColorChangeGlobalRegistrationDebug(viewport, newColor, idx):
    # For GlobalRegistered cloud
    cloudName = f"cloud_globalregistered_{idx}"

    # Create new color for the material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = (newColor.red, newColor.green, newColor.blue, 1.0)

    # Update material in viewport
    viewport.scene.modify_geometry_material(cloudName, material)

def onColorChangeRegistrationDebug(viewport, newColor, idx):
    # For Registered cloud
    cloudName = f"cloud_registered_{idx}"

    # Create new color for the material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = (newColor.red, newColor.green, newColor.blue, 1.0)

    # Update material in viewport
    viewport.scene.modify_geometry_material(cloudName, material)

#-----------------------------
# Checkbox Configurations
#-----------------------------
def onCheckboxShowOriginalToggled(viewport, idx, isChecked):
    # For original cloud
    cloudName = f"cloud_{idx}"

    if isChecked:
        viewport.scene.show_geometry(cloudName, True)   # Show Again  
    else:
        viewport.scene.show_geometry(cloudName, False)  # Hide 

def onCheckboxShowGlobalRegistrationDebugToggled(viewport, idx, isChecked):
    # For GlobalRegistered cloud
    cloudName = f"cloud_globalregistered_{idx}"

    if isChecked:
        viewport.scene.show_geometry(cloudName, True)   # Show Again  
    else:
        viewport.scene.show_geometry(cloudName, False)  # Hide 

def onCheckboxShowRegistrationDebugToggled(viewport, idx, isChecked):
    # For Registered cloud
    cloudName = f"cloud_registered_{idx}"

    if isChecked:
        viewport.scene.show_geometry(cloudName, True)   # Show Again  
    else:
        viewport.scene.show_geometry(cloudName, False)  # Hide 


#-----------------------------
# Viewport Configuration
#-----------------------------
def viewportConfiguration(window, pointClouds, pointCloudsGlobalRegistrationList, pointCloudsRegistrationDebug):
    # Create the Viewport
    viewport = o3d.visualization.gui.SceneWidget()
    viewport.scene = o3d.visualization.rendering.Open3DScene(window.renderer)

    # Create new color for the material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    # Call the pointClouds
    for idx, pointCloud in enumerate(pointClouds):
        viewport.scene.add_geometry(f"cloud_{idx}", pointCloud, material)
        viewport.scene.show_geometry(f"cloud_{idx}", False)  # Hide 

    # Call the pointClouds GlobalRegistered
    for idx, pointCloud in enumerate(pointCloudsRegistrationDebug):
        viewport.scene.add_geometry(f"cloud_registered_{idx}", pointCloud, material)
        viewport.scene.show_geometry(f"cloud_registered_{idx}", False)  # Hide 

    # Call the pointClouds Registered
    for idx, pointCloud in enumerate(pointCloudsGlobalRegistrationList):
        viewport.scene.add_geometry(f"cloud_globalregistered_{idx}", pointCloud, material)
        viewport.scene.show_geometry(f"cloud_globalregistered_{idx}", False)  # Hide 
    
    # Setup camera to start with a good viewpoint 
    if pointClouds:
        bounds = pointClouds[0].get_axis_aligned_bounding_box()
        for pointCloud in pointClouds[1:]:
                bounds += pointCloud.get_axis_aligned_bounding_box()
                viewport.setup_camera(60, bounds, bounds.get_center())

        viewport.background_color = o3d.visualization.gui.Color(1, 1, 1)

    return viewport


#-----------------------------
# controlPanel Configuration
#-----------------------------
def controlPanelConfiguration(window, pointClouds, viewport):
    # ControlPanel Configuration
    controlPanel = o3d.visualization.gui.Vert(0.25 * window.theme.font_size,
        o3d.visualization.gui.Margins(10, 10, 10, 10))

    # Create and activate checkboxes/label 
    for idx, _ in enumerate(pointClouds):
        # create a exception for the first(target) pointCloud

        #Create a label for each pcd and add the color name
        label = o3d.visualization.gui.Label(f"Point Cloud {idx}")

        # Create checkboxes
        checkboxShowOriginal = o3d.visualization.gui.Checkbox(f"Show PointCloud Original")
        checkboxShowGlobalRegistrationDebug = o3d.visualization.gui.Checkbox(f"Show PointCloud Global Registered")
        checkboxShowRegistrationDebug = o3d.visualization.gui.Checkbox(f"Show PointCloud Registered")

        # Create ColorPickers
        colorPickerOriginal = o3d.visualization.gui.ColorEdit()
        colorPickerGlobalRegistrationDebug = o3d.visualization.gui.ColorEdit()
        colorPickerRegistrationDebug = o3d.visualization.gui.ColorEdit()

        # Link checkboxes
        checkboxShowOriginal.set_on_checked(partial(onCheckboxShowOriginalToggled, viewport, idx))
        checkboxShowGlobalRegistrationDebug.set_on_checked(partial(onCheckboxShowGlobalRegistrationDebugToggled, viewport, idx))
        checkboxShowRegistrationDebug.set_on_checked(partial(onCheckboxShowRegistrationDebugToggled, viewport, idx))

        # Link ColorPickers
        colorPickerOriginal.set_on_value_changed(partial(onColorChange, viewport, idx = idx))
        colorPickerGlobalRegistrationDebug.set_on_value_changed(partial(onColorChangeGlobalRegistrationDebug, viewport, idx = idx))
        colorPickerRegistrationDebug.set_on_value_changed(partial(onColorChangeRegistrationDebug, viewport, idx = idx))

        # Add all to control Panel 
        controlPanel.add_child(label)
        controlPanel.add_child(checkboxShowOriginal)
        controlPanel.add_child(colorPickerOriginal)

        if idx > 0:
            controlPanel.add_child(checkboxShowGlobalRegistrationDebug)
            controlPanel.add_child(colorPickerGlobalRegistrationDebug)
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
    # Calculate Global Regist Distance Threshold
    ransacDistanceThreshold = args['voxelSize'] * 3
    # Calculate Global Regist radiusFeature
    radiusFeature = args['voxelSize'] * 5

    # Compute features (fpfh)
    pointCloudFeatures = compute_fpfh_feature(pointCloudDownsampled,
        KDTreeSearchParamHybrid(radius=radiusFeature, max_nn=100))
    accumulatedPointCloudFeatures = compute_fpfh_feature(accumulatedPointCloudDownsampled,
        KDTreeSearchParamHybrid(radius=radiusFeature, max_nn=100))

    # Compute globalRegistration transformation
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


def main():

    # -----------------------------
    #  Parse arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Read Poit Clouds To Register")
    parser.add_argument('-ifd', "--inputFolder",  default='../image_to_cloud/output_clouds', type=str,
        help="Folder where the pointclouds images are.)")
    parser.add_argument('-outpcd', "--outputPointClouds", default='cloud_registered', type=str,
        help="Folder to save the pcd)")
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
    # Create a dictionary to put all the statistics from GlobalRegistration and ICP
    registeredTable = {} #only for debug table

    # Create a deepcopy of my pointclouds to apply the transformations
    pointCloudsRegistrationDebug = copy.deepcopy(pointClouds)

    # Create the point cloud witch will be fixed (accumulated)
    accumulatedPointCloud = o3d.geometry.PointCloud()

    # Create a downsample version to save 
    accumulatedPointCloudDownsampled = o3d.geometry.PointCloud()

    # Create a list to have all the pointClouds only globalRegistered
    pointCloudsGlobalRegistrationList = []


    # Create a for loop to apply the transformation to all the sources(the ones who will move)
    for idx, pointCloud in enumerate(pointCloudsRegistrationDebug):
        
        #Create a list with range idx
        numberPC = list(range(idx))
        #Create a string with - (0-1-2...)
        listText = " - ".join(str(number) for number in numberPC)
        
        # Create a if for the first pointCloud that arrives
        if len(accumulatedPointCloud.points) == 0: # no points in the accumulated
            
            # Transformation is identity for the firstCloud(globalregist to visualize and final)
            estimatedTransformation = np.eye(4) 
            ransacEstimatedTransformation = np.eye(4)

            # Downsample and Estimate Normals
            pointCloudDownsampled = downsampleAndEstimateNormals(pointCloud, args)

        else:

            # Downsample and Estimate Normals
            pointCloudDownsampled = downsampleAndEstimateNormals(pointCloud, args)

            # Global Registration procedure 
            globalRegistrationTransformation = calculateGlobalRegistrationTransformation(accumulatedPointCloudDownsampled, pointCloudDownsampled,  args)
            print(f"[RANSAC] [{listText}] -- {idx} done")

            ransacEstimatedTransformation = globalRegistrationTransformation.transformation
            
            # ICP Registration procedure (call the CustomICP class)
            customICP = CustomICP()
            estimatedTransformation, rmse = customICP.run(pointCloud, accumulatedPointCloud, globalRegistrationTransformation)

        # -----------------------------------------
        # Apply Ransac Transformation for visualization
        # -----------------------------------------
        pointCloudGlobalRegistrationDebug = copy.deepcopy(pointCloud)
        pointCloudGlobalRegistration = pointCloudGlobalRegistrationDebug.transform(
        ransacEstimatedTransformation
        )
        pointCloudsGlobalRegistrationList.append(pointCloudGlobalRegistration)
        
        # -----------------------------------------
        # Apply Final Transformation and accumulate
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
                f"{globalRegistrationTransformation.inlier_rmse:.4f}",
                f"{rmse:.4f}"
            ]
            
            # Save the values in a dictionary
            registeredTable[f"[{listText}] -- {idx}"] = registeredValues

            # Print accumulated points
            print(f'Points in the accumulated cloud [{listText}] -- {idx}: ' + str(len(accumulatedPointCloud.points)))


    # -----------------------------------------
    # Table for print
    # -----------------------------------------
    print(f"{'Point Clouds':<25} | {'RANSAC RMSE':<16} | {'ICP_RMSE':<16}")
    print("-" * 59)
    for key, values in registeredTable.items():
        print(f"{key:<25} | {values[0]:<16} | {values[1]:<16}")


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

    # Call viewport
    viewport = viewportConfiguration(window, pointClouds, pointCloudsGlobalRegistrationList, pointCloudsRegistrationDebug)

    # Call control panel
    controlPanel = controlPanelConfiguration(window, pointClouds, viewport)

    # Add viewport and controlPanel to the window
    window.add_child(viewport)
    window.add_child(controlPanel)

    # Handles viewport + control panel positioning
    window.set_on_layout(partial(onLayout, window, controlPanel, viewport))

    # Run the app
    application.run()

if __name__ == '__main__':
    main()
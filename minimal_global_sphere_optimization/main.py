#!/usr/bin/env python3
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from functools import partial
import threading
import time
from open3d.visualization import gui, rendering

#-----------------------------
# Layout Configurations
#-----------------------------
def onLayout(window, controlPanel, viewport, ctx):
    # Rectangle size
    rectangleSize = window.content_rect

    # Layout and scene frame
    controlPanel.frame = o3d.visualization.gui.Rect(rectangleSize.width - 210, 20, 210, rectangleSize.height - 80)
    viewport.frame = o3d.visualization.gui.Rect(rectangleSize.x, rectangleSize.y, rectangleSize.width, rectangleSize.height)


# ------------------------------------------------------------
# Material for transparent sphere
# ------------------------------------------------------------
def createSphereMaterial():
    material = rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [1.0, 0.0, 0.0, 0.25]   # RGBA
    return material


# ------------------------------------------------------------
# GUI setup (Viewport + Open3DScene)
# ------------------------------------------------------------
def configureViewport(window, pointCloud):
    sceneWidget = gui.SceneWidget()
    sceneWidget.scene = rendering.Open3DScene(window.renderer)

    # Point cloud material
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    sceneWidget.scene.add_geometry("pointCloud", pointCloud, material)

    bounds = pointCloud.get_axis_aligned_bounding_box()
    sceneWidget.setup_camera(60.0, bounds, bounds.get_center())
    sceneWidget.background_color = gui.Color(1, 1, 1)

    window.add_child(sceneWidget)
    return sceneWidget


#-----------------------------
# Paint Point Clouds Configurations
#-----------------------------
def onColorChange(viewport, newColor):
    # For original cloud
    cloudName = "pointCloud"

    # Create new color for the material
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.base_color = (newColor.red, newColor.green, newColor.blue, 1.0)

    # Update material in viewport
    viewport.scene.modify_geometry_material(cloudName, material)


#-----------------------------
# controlPanel Configuration
#-----------------------------
def controlPanelConfiguration(window, pointClouds, viewport):
    # ControlPanel Configuration
    controlPanel = o3d.visualization.gui.Vert(0.25 * window.theme.font_size,
        o3d.visualization.gui.Margins(10, 10, 10, 10))

    #Create a label for each pcd and add the color name
    label = o3d.visualization.gui.Label("Point Cloud")

    # Create ColorPickers
    colorPickerOriginal = o3d.visualization.gui.ColorEdit()

    # Link ColorPickers
    colorPickerOriginal.set_on_value_changed(partial(onColorChange, viewport))

    # Add all to control Panel 
    controlPanel.add_child(label)
    controlPanel.add_child(colorPickerOriginal)

    return controlPanel


# ------------------------------------------------------------
# Objective function (SLSQP minimizes the radius)
# ------------------------------------------------------------
def objectiveFunction(params):
    return abs(params[3])     # radius is parameter 3 (never negative)


# ------------------------------------------------------------
# Constraint: distance(point_i, center) <= radius
# ------------------------------------------------------------
def constraintFunction(params, points):
    center = np.array(params[:3])
    radius = params[3]

    distances = np.linalg.norm(points - center, axis=1)
    return radius - distances     # must be >= 0 because ineq


# ------------------------------------------------------------
# Callback - animate sphere each iteration
# ------------------------------------------------------------
def iterationCallback(params, window, viewport):
    center = np.array(params[:3])
    radius = params[3]

    sphereMesh = createTransparentSphere(center, radius)
    updateSphereInGui(window, viewport, sphereMesh)

    time.sleep(1)     # Animation speed


# ------------------------------------------------------------
# Create transparent sphere mesh
# ------------------------------------------------------------
def createTransparentSphere(center, radius):
    radius = max(abs(radius), 1e-6)   #some values given by the SLSQP callback re inappropriate to make a sphere (raio nunca é negativo nem 0)

    mesh = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=20)
    mesh.compute_vertex_normals()
    mesh.translate(center)
    return mesh


# ------------------------------------------------------------
# Update the sphere in the GUI (runs on GUI thread)
# ------------------------------------------------------------
def updateSphereInGui(window, sceneWidget, sphereMesh):
    sphereMaterial = createSphereMaterial()

    def update():
        if sceneWidget.scene.has_geometry("sphere"):
            sceneWidget.scene.remove_geometry("sphere")
        sceneWidget.scene.add_geometry("sphere", sphereMesh, sphereMaterial)


    # Update function in stanby until ir updates
    gui.Application.instance.post_to_main_thread(window, update)


# ------------------------------------------------------------
# Optimization thread (animation handled in callback)
# ------------------------------------------------------------
def optimizationThread(points, initialParams, window, viewport):

    # Build constraints with partial
    constraints = {
        "type": "ineq",
        "fun": partial(constraintFunction, points=points)
    }

    # Build callback with partial
    animatedCallback = partial(
        iterationCallback,
        window=window,
        viewport=viewport,
    )

    # ------------------------------------------------------------
    # Run Minimize
    # ------------------------------------------------------------
    result = minimize(
        fun=objectiveFunction,
        x0=initialParams,
        method="SLSQP",
        constraints=constraints,
        callback=animatedCallback,
        options={"maxiter": 200, "ftol": 1e-12, "disp": True}
    )

    # Print results
    finalCenter = np.array(result.x[:3])
    finalRadius = abs(result.x[3])

    print("\nOptimization Complete")
    print("Final Center:", finalCenter)
    print("Final Radius:", finalRadius)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    # Load point cloud
    pointCloud = o3d.io.read_point_cloud("pcd_to_work/cloud_registered.pcd")
    pointsNp = np.asarray(pointCloud.points)

    # Initial guess: center and radius
    #center0 = pointsNp.mean(axis=0)
    #radius0 = np.max(np.linalg.norm(pointsNp - center0, axis=1))
    #initialParams = np.array([center0[0], center0[1], center0[2], radius0])

    # Initialize GUI
    app = gui.Application.instance
    app.initialize()

    window = app.create_window("Minimum Enclosing Sphere Optimizer", 1280, 800)

    # Call viewport
    viewport = configureViewport(window, pointCloud)

    # Call control panel
    controlPanel = controlPanelConfiguration(window, pointCloud, viewport)

    # Add viewport and controlPanel to the window
    window.add_child(viewport)
    window.add_child(controlPanel)

    # Handles viewport + control panel positioning
    window.set_on_layout(partial(onLayout, window, controlPanel, viewport))

    # Start optimization thread
    thread = threading.Thread(
        target=optimizationThread,
        args=(pointsNp, np.zeros(4), window, viewport),
        daemon=True
    )
    thread.start()

    # Run GUI loop
    app.run()


if __name__ == "__main__":
    main()

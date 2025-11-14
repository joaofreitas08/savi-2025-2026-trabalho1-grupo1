#!/usr/bin/env python3
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from functools import partial
import threading
import time

from open3d.visualization import gui, rendering


# ------------------------------------------------------------
# Objective function (SLSQP minimizes the radius)
# ------------------------------------------------------------
def objectiveFunction(params):
    return abs(params[3])     # radius is parameter 3


# ------------------------------------------------------------
# Constraint: distance(point_i, center) <= radius
# ------------------------------------------------------------
def constraintFunction(params, points):
    center = np.array(params[:3])
    radius = params[3]

    distances = np.linalg.norm(points - center, axis=1)
    return radius - distances      # must be >= 0


# ------------------------------------------------------------
# CALLBACK: ANIMATE SPHERE EACH ITERATION
# ------------------------------------------------------------
def iterationCallback(params, window, sceneWidget):
    center = np.array(params[:3])
    radius = params[3]

    sphereMesh = createTransparentSphere(center, radius)
    updateSphereInGui(window, sceneWidget, sphereMesh)

    time.sleep(0.5)     # Animation speed


# ------------------------------------------------------------
# Create transparent sphere mesh
# ------------------------------------------------------------
def createTransparentSphere(center, radius):
    radius = max(abs(radius), 1e-6)   # <--- PROTECTION AGAINST R <= 0

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
# Material for transparent sphere
# ------------------------------------------------------------
def createSphereMaterial():
    material = rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [1.0, 0.0, 0.0, 0.25]   # RGBA
    return material


# ------------------------------------------------------------
# GUI setup (SceneWidget + Open3DScene)
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


# ------------------------------------------------------------
# Optimization thread (animation handled in callback)
# ------------------------------------------------------------
def optimizationThread(points, initialParams, window, sceneWidget):

    # -------------------------------
    # Build constraints
    # -------------------------------
    constraints = {
        "type": "ineq",
        "fun": partial(constraintFunction, points=points)
    }

    
    # Build callback with partial
    animatedCallback = partial(
        iterationCallback,
        window=window,
        sceneWidget=sceneWidget,
    )

    # ------------------------------------------------------------
    # Run SLSQP
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
    sceneWidget = configureViewport(window, pointCloud)

    # Start optimization thread
    thread = threading.Thread(
        target=optimizationThread,
        args=(pointsNp, np.zeros(4), window, sceneWidget),
        daemon=True
    )
    thread.start()

    # Run GUI loop
    app.run()


if __name__ == "__main__":
    main()

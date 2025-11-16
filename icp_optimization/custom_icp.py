import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import copy
from functools import partial
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import time


class CustomICP:
    # -----------------------------------------
    # Initialize Class
    # -----------------------------------------
    def __init__(self, verbose=True):

        # Initialize main ICP parameters
        self.verbose = verbose                                  # print debug info if True
        self.voxelSize = 0.05                                   # define the vexelSize
        self.distanceThreshold = self.voxelSize * 1.5           # define the distanceThreshold
        self.distancefScale = self.voxelSize * 0.5              # define the fScale for huber loss
        self.maxIcpIterations = 100                             # define the number of max iterations
        self.icpConvergenceTolerance = 1e-6                     # define the tolerance for icp


    # -----------------------------------------
    # Visualzer for each Iteration 
    # -----------------------------------------
    def initVisualizer(self, targetCloud):

        # Create the visualizer 
        self.visualizer = o3d.visualization.Visualizer()

        # Create the visualizer window
        self.visualizer.create_window("ICP Animation", width=1500, height=800) #800/600

        # Create the source pointCloud to update
        self.sourceVisualization = o3d.geometry.PointCloud()

        # Make deep copy of the target pointCloud
        self.targetVisualization = copy.deepcopy(targetCloud)

        # Paint the target pointCloud
        self.targetVisualization.paint_uniform_color([0, 1, 0])   # green

        # Add both pointClouds
        self.visualizer.add_geometry(self.sourceVisualization)
        self.visualizer.add_geometry(self.targetVisualization)

        # Define the pointSize
        renderOption = self.visualizer.get_render_option()
        renderOption.point_size = 4.0    

        # Camera Definitions
        view = self.visualizer.get_view_control()
        # Camera bounds
        bounds = self.targetVisualization.get_axis_aligned_bounding_box()

        # Camera center
        center = bounds.get_center()
        view.set_lookat(center)

        # Camera Rotation
        view.set_front([0, 0, -1])  
        view.set_up([0, -1, 0])    

        view.set_zoom(0.4)    

        # Update
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

   
    # -----------------------------------------
    # Add a callback that applies the current ICP update and renders
    # -----------------------------------------
    def iterationCallback(self, x, transformedSourcePoints):

        # X is the vector of updated values
        transformation = self.transformationMatrix(x)

        # Update point cloud with the values
        updatedPointCloud = self.transformPoints(transformedSourcePoints, transformation)

        # Transform the poin cloud in a o3d pcd
        updatedPointCloudO3D = o3d.utility.Vector3dVector(updatedPointCloud)

        # Paint the point CLoud
        self.sourceVisualization.points = updatedPointCloudO3D
        self.sourceVisualization.paint_uniform_color([1, 0, 0])   # red

        # Update
        self.visualizer.update_geometry(self.sourceVisualization)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

        time.sleep(0.1)


    # -----------------------------------------
    # Apply a 4x4 transformation matrix to Nx3 points
    # -----------------------------------------
    @staticmethod
    def transformPoints(sourcePoints, transformation):

        # Convert points to homogeneous coordinates (N, 4)
        pointsHomogeneous = np.hstack((sourcePoints, np.ones((sourcePoints.shape[0], 1))))

        # Apply transformation and transpose back to (N, 4)
        transformedHomogeneous = (transformation @ pointsHomogeneous.T).T

        # Return transformed 3D points (N, 3)
        return transformedHomogeneous[:, :3]


    # -----------------------------------------
    # Convert 6 motion parameters into a 4×4 transformation matrix
    # -----------------------------------------
    @staticmethod
    def transformationMatrix(parameters):

        rX, rY, rZ, tX, tY, tZ = parameters

        # Extract the rotation angles (in radians)
        # These represent small rotations around X, Y, Z
        rotation_angles = [rX, rY, rZ]

        # Convert Euler angles into a rotation matrix
        rotation = R.from_euler('xyz', rotation_angles)

        # Get the 3×3 rotation matrix
        combinedRotation = rotation.as_matrix()

        # Construct the full 4×4 transformation matrix
        transformationMatrix = np.eye(4)
        transformationMatrix[:3, :3] = combinedRotation     # rotation block
        transformationMatrix[:3, 3] = [tX, tY, tZ]          # translation vector

        return transformationMatrix


    # -----------------------------------------
    # Find nearest-neighbor correspondences using a KDTree
    # -----------------------------------------
    def sourceToTargetCorrespondencesIndex(self, sourceCloud):

        # Query the precomputed KDTree of the target cloud.
        # For each point in the source cloud, find the index of the closest point in the target cloud.    
        _, sourceToTargetCorrespondencesIndex = self.kdTree.query(sourceCloud, k=1)

        # Return the correspondence indices as a NumPy array.
        return np.array(sourceToTargetCorrespondencesIndex)
    

    # -----------------------------------------
    # Compute residuals between matched source and target points
    # -----------------------------------------
    def objectiveFunction(self, parameters, transformedSourcePoints, matchedTargetPoints):
        
        # Convert optimization parameters (rX, rY, rZ, tX, tY, tZ) into a 4x4 incremental transformation matrix.
        deltaTransformation = self.transformationMatrix(parameters)  # 4x4 matrix

        # Apply the incremental transformation to the source points.
        transformedSourcePoints = self.transformPoints(transformedSourcePoints, deltaTransformation)  # (N, 3)

        # Compute residuals (differences) between transformed source points and their corresponding target points.
        differences = matchedTargetPoints - transformedSourcePoints   # (N, 3)

        # Flatten the residual matrix into a 1D array (required by scipy.optimize.least_squares).
        return differences.ravel()


    # -----------------------------------------
    # Main ICP optimization loop
    # -----------------------------------------
    def run(self, sourceCloud, targetCloud, globalRegistrationTransformation):

        # Downsample pointCLouds
        sourceCloudDownsampled = sourceCloud.voxel_down_sample(self.voxelSize)
        targetCloudDownsampled = targetCloud.voxel_down_sample(self.voxelSize)

        # Convert Open3D point clouds to NumPy arrays
        sourcePoints = np.asarray(sourceCloudDownsampled.points)
        targetPoints = np.asarray(targetCloudDownsampled.points)  # Target cloud stays fixed

        # Compute the kdTree for the targetCloud   
        self.kdTree = cKDTree(np.asarray(targetCloudDownsampled.points)) 

        # Copy the first transformation given by globalRegistration
        currentTransformation = globalRegistrationTransformation.transformation.copy()

        #Initialize the viewer
        self.initVisualizer(targetCloudDownsampled)

        # Loop for (maxICPIterations)
        for iteration in range(self.maxIcpIterations):
       
            # Transform the source cloud with the current transformation
            transformedSourcePoints = self.transformPoints(sourcePoints, currentTransformation)

            # Find nearest-neighbor correspondences (source to target)
            sourceToTargetCorrespondencesIndex = self.sourceToTargetCorrespondencesIndex(transformedSourcePoints)

            # Retrieve matched target points using the correspondence indices
            matchedTargetPoints = targetPoints[sourceToTargetCorrespondencesIndex]

            # Calculate distances to apply mask
            distances = np.linalg.norm(matchedTargetPoints - transformedSourcePoints, axis=1)

            # Create the mask
            mask = distances < self.distanceThreshold

            # Keep only inliers
            # When rejecting an outlier, we must remove the entire pair;
            # otherwise source and target would become misaligned (different sizes or wrong point pairs).
            transformedSourcePoints = transformedSourcePoints[mask]
            matchedTargetPoints = matchedTargetPoints[mask]

            # Define objective (residual) function for least squares optimization
            objectiveFunction = partial(
                    self.objectiveFunction,
                    transformedSourcePoints=transformedSourcePoints,
                    matchedTargetPoints=matchedTargetPoints,
            )

            # Solve for the incremental transformation using robust least squares
            leastSquaresResult = least_squares(
                    objectiveFunction,
                    np.zeros(6),                                                                # Initial parameters: [rX, rY, rZ, tX, tY, tZ]
                    method='trf',                                                               # Trust Region Reflective method (supports robust loss)           
                    loss='huber',                                                               # Huber loss 
                    f_scale=self.distancefScale,                                                # Scale defining inlier region for Huber loss
                    verbose=0,
                    callback= partial(self.iterationCallback, 
                                      transformedSourcePoints=transformedSourcePoints)          # Internal solver output for visualization and debug
            )

            deltaTransformation = self.transformationMatrix(leastSquaresResult.x)

            # Update pose
            currentTransformation = deltaTransformation @ currentTransformation
            
            print(f"Iteration {iteration}")

            # Convergence check
            # leastSquaresResult.x = if the euclidian norm of the values form leastsquares given parameters < defined tolerence break
            if np.linalg.norm(leastSquaresResult.x) < self.icpConvergenceTolerance:
                if self.verbose:
                    print(f"ICP converged at iteration {iteration}")
                break

        # Close window when optimization ends
        self.visualizer.destroy_window()

        #print(leastSquaresResult.fun)
        # Final RMSE    
        rootMeanSquaredError = np.sqrt(np.mean(leastSquaresResult.fun ** 2))


        return currentTransformation, rootMeanSquaredError
         

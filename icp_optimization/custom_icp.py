import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import copy
from functools import partial
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
import time


class CustomICP:
    """Simplified Point-to-Point ICP implementation using SciPy least_squares."""

    def __init__(self, verbose=True):
        # -----------------------------------------
        # Initialize main ICP parameters
        # -----------------------------------------
        self.verbose = verbose                                  # print debug info if True
        self.finalTransform = np.eye(4)                         # final 4x4 transformation matrix
        self.voxelSize = 0.01
        self.distanceThreshold = self.voxelSize * 1.5          # define the distanceThreshold


    # -----------------------------------------
    # Visualzer for each Iteration 
    # -----------------------------------------
    def init_visualizer(self, targetCloud):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("ICP Animation", width=800, height=600)

        
        self.sourceVisualization = o3d.geometry.PointCloud()
        # Make deep copies so original clouds stay intact
        self.targetVisualization = copy.deepcopy(targetCloud)

        
        self.targetVisualization.paint_uniform_color([0, 1, 0])   # green

        self.vis.add_geometry(self.sourceVisualization)
        self.vis.add_geometry(self.targetVisualization)
        
        renderOption = self.vis.get_render_option()
        renderOption.point_size = 2.0        


        self.vis.poll_events()
        self.vis.update_renderer()

    # -----------------------------------------
    # Add a callback that applies the current ICP update and rerenders 
    # -----------------------------------------
    def iterationCallback(self, x, transformedSourcePoints):
        # x é o vetor de parâmetros atual da iteração
        transformation = self.smallTransform(x)

        updatedPointCloud = self.transformPoints(transformedSourcePoints, transformation)

        updatedPointCloudO3D = o3d.utility.Vector3dVector(updatedPointCloud)

        self.sourceVisualization.points = updatedPointCloudO3D
        self.sourceVisualization.paint_uniform_color([1, 0, 0])   # red

        self.vis.update_geometry(self.sourceVisualization)
        self.vis.poll_events()
        self.vis.update_renderer()

        time.sleep(0.01)


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
    # Convert 6 motion parameters into a 4x4 transformation
    # -----------------------------------------
    @staticmethod
    def smallTransform(parameters):
        rX, rY, rZ, tX, tY, tZ = parameters

        # Build rotation matrices
        rotationX = np.array([
            [1, 0, 0],
            [0, np.cos(rX), -np.sin(rX)],
            [0, np.sin(rX),  np.cos(rX)]
        ])
        rotationY = np.array([
            [np.cos(rY), 0, np.sin(rY)],
            [0, 1, 0],
            [-np.sin(rY), 0, np.cos(rY)]
        ])
        rotationZ = np.array([
            [np.cos(rZ), -np.sin(rZ), 0],
            [np.sin(rZ),  np.cos(rZ), 0],
            [0, 0, 1]
        ])

        # Combine rotations (Z-Y-X) and add translation
        combinedRotation = rotationZ @ rotationY @ rotationX
        transformationMatrix = np.eye(4)
        transformationMatrix[:3, :3] = combinedRotation
        transformationMatrix[:3, 3] = [tX, tY, tZ]

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
    def objectiveFunction(self, parameters, transformedSourcePoints, targetPoints):
        
        # Convert optimization parameters (rX, rY, rZ, tX, tY, tZ) into a 4x4 incremental transformation matrix.
        deltaTransformation = self.smallTransform(parameters)  # 4x4 matrix

        # Apply the incremental transformation to the source points.
        transformedSourcePoints = self.transformPoints(transformedSourcePoints, deltaTransformation)  # (N, 3)

        # Find nearest-neighbor correspondences (source → target)
        sourceToTargetCorrespondencesIndex = self.sourceToTargetCorrespondencesIndex(transformedSourcePoints)

        # Retrieve matched target points using the correspondence indices
        matchedTargetPoints = targetPoints[sourceToTargetCorrespondencesIndex]

        # Compute residuals (differences) between transformed source points and their corresponding target points.
        differences = matchedTargetPoints - transformedSourcePoints   # (N, 3)

        # Flatten the residual matrix into a 1D array (required by scipy.optimize.least_squares).
        return differences.ravel()

    # -----------------------------------------
    # Main ICP optimization loop
    # -----------------------------------------
    def run(self, sourceCloud, targetCloud, globalRegistrationTransformation):
        # Convert Open3D point clouds to NumPy arrays
        sourcePoints = np.asarray(sourceCloud.points)
        targetPoints = np.asarray(targetCloud.points)  # Target cloud stays fixed

        # Copy the first transformation given by globalRegistration
        firstTransformation = globalRegistrationTransformation.transformation.copy()

        # Compute the kdTree for the targetCloud   
        self.kdTree = cKDTree(np.asarray(targetCloud.points))   
       
        # Transform the source cloud with the current transformation
        transformedSourcePoints = self.transformPoints(sourcePoints, firstTransformation)

        #Initialize the viewer
        self.init_visualizer(targetCloud)


        # Define objective (residual) function for least squares optimization
        objectiveFunction = partial(
                self.objectiveFunction,
                transformedSourcePoints=transformedSourcePoints,
                targetPoints=targetPoints,
        )

        # Solve for the incremental transformation using robust least squares
        residualResultLeastSquares = least_squares(
                objectiveFunction,
                np.zeros(6),                        # Initial parameters: [rX, rY, rZ, tX, tY, tZ]
                method='trf',                       # Trust Region Reflective method (supports robust loss)
                ftol=1e-05,
                loss='huber',                       # Robust loss function to reduce outlier influence // linear rho(z) = z if z <= 1 else 2*z**0.5 - 1 quatratic
                f_scale=self.distanceThreshold,     # Scale defining inlier region for Huber loss
                verbose=2,
                callback= partial(self.iterationCallback, transformedSourcePoints=transformedSourcePoints)                      # Internal solver output for debugging
        )

        # close window when optimization ends
        self.vis.destroy_window() 

        # Compute RMSE (Root Mean Square Error) of residuals
        rootMeanSquaredError = np.sqrt(np.mean(residualResultLeastSquares.fun ** 2))

        # Convert optimized parameters into a 4x4 transformation matrix
        resultLeastSquaresTransformation = self.smallTransform(residualResultLeastSquares.x)

        # Create the final transformation
        finalTransformation = resultLeastSquaresTransformation @ firstTransformation


        # Report iteration results
        if self.verbose:
            print(f"RMSE error = {rootMeanSquaredError:.6f}")

        return finalTransformation, rootMeanSquaredError


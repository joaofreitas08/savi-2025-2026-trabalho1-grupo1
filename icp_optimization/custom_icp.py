import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import copy
from functools import partial
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree

class CustomICP:
    """Simplified Point-to-Point ICP implementation using SciPy least_squares."""

    def __init__(self, maxIterations, tolerance, targetCloud, verbose=True):
        # -----------------------------------------
        # Initialize main ICP parameters
        # -----------------------------------------
        self.maxIterations = maxIterations                      # maximum number of ICP iterations
        self.tolerance = tolerance                              # convergence threshold
        self.verbose = verbose                                  # print debug info if True
        self.finalTransform = np.eye(4)                         # final 4x4 transformation matrix
        self.voxelSize = 0.05
        self.distanceThreshold = self.voxelSize * 10            # define the distanceThreshold
        self.kdTree = cKDTree(np.asarray(targetCloud.points))   # compute the kdTree for the tagetCloud   


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
    def objectiveFunction(self, parameters, transformedSourcePoints, matchedTargetPoints):
        # Convert optimization parameters (rX, rY, rZ, tX, tY, tZ) into a 4x4 incremental transformation matrix.
        deltaTransformation = self.smallTransform(parameters)  # 4x4 matrix

        # Apply the incremental transformation to the source points.
        transformedSourcePoints = self.transformPoints(transformedSourcePoints, deltaTransformation)  # (N, 3)

        # Compute residuals (differences) between transformed source points and their corresponding target points.
        differences = transformedSourcePoints - matchedTargetPoints  # (N, 3)

        # Flatten the residual matrix into a 1D array (required by scipy.optimize.least_squares).
        return differences.ravel()


    # -----------------------------------------
    # Main ICP optimization loop
    # -----------------------------------------
    def run(self, sourceCloud, targetCloud, globalRegistrationTransformation):
        # Convert Open3D point clouds to NumPy arrays
        sourcePoints = np.asarray(sourceCloud.points)
        targetPoints = np.asarray(targetCloud.points)  # Target cloud stays fixed

        # Initialize transformation from global registration
        currentTransformation = copy.deepcopy(globalRegistrationTransformation.transformation)

        # Main ICP iteration loop
        for i in range(self.maxIterations):

            # Transform the source cloud with the current transformation
            transformedSourcePoints = self.transformPoints(sourcePoints, currentTransformation)

            # Find nearest-neighbor correspondences (source → target)
            sourceToTargetCorrespondencesIndex = self.sourceToTargetCorrespondencesIndex(transformedSourcePoints)

            # Retrieve matched target points using the correspondence indices
            matchedTargetPoints = targetPoints[sourceToTargetCorrespondencesIndex]

            # Define objective (residual) function for least squares optimization
            objectiveFunction = partial(
                self.objectiveFunction,
                transformedSourcePoints=transformedSourcePoints,
                matchedTargetPoints=matchedTargetPoints,
            )

            # Solve for the incremental transformation using robust least squares
            residualResultLeastSquares = least_squares(
                objectiveFunction,
                np.zeros(6),                        # Initial parameters: [rX, rY, rZ, tX, tY, tZ]
                method='trf',                       # Trust Region Reflective method (supports robust loss)
                loss='huber',                       # Robust loss function to reduce outlier influence // rho(z) = z if z <= 1 else 2*z**0.5 - 1
                f_scale=self.distanceThreshold,     # Scale defining inlier region for Huber loss
                verbose=2                           # Internal solver output for debugging
            )

            # Compute RMSE (Root Mean Square Error) of residuals
            rootMeanSquaredError = np.sqrt(np.mean(residualResultLeastSquares.fun ** 2))

            # Convert optimized parameters into a 4x4 transformation matrix
            resultLeastSquaresTransformation = self.smallTransform(residualResultLeastSquares.x)

            # Update the current transformation (compose incrementally)
            currentTransformation = resultLeastSquaresTransformation @ currentTransformation
            # print the current transformation matrix
            print(currentTransformation)

            # Report iteration results
            if self.verbose:
                print(f"Iteration {i + 1:02d}: error = {rootMeanSquaredError:.6f}")

            # Check convergence condition
            if np.linalg.norm(residualResultLeastSquares.x) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {i + 1}")
                break

        # -----------------------------------------
        # Store and return final results
        # -----------------------------------------
        self.finalTransform = copy.deepcopy(currentTransformation)
        return self.finalTransform, rootMeanSquaredError


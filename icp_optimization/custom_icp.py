import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import copy
from functools import partial

class CustomICP:
    """Simplified Point-to-Point ICP implementation using SciPy least_squares."""

    def __init__(self, maxIterations, tolerance, verbose=True):
        # -----------------------------------------
        # Initialize main ICP parameters
        # -----------------------------------------
        self.maxIterations = maxIterations   # maximum number of ICP iterations
        self.tolerance = tolerance           # convergence threshold
        self.verbose = verbose               # print debug info if True
        self.finalTransform = np.eye(4)      # final 4x4 transformation matrix
        self.errors = [] 
        self.voxelSize = 0.05
        self.distanceThreshold = self.voxelSize * 1.5                 # store per-iteration mean squared error

    # -----------------------------------------
    # Apply a 4x4 transformation matrix to Nx3 points
    # -----------------------------------------
    @staticmethod
    def transformPoints(points, transform):
        """
        Applies a 4x4 homogeneous transformation to 3D points.
        """
        # Convert points to homogeneous coordinates (N,4)
        pointsHomogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        # Apply the transformation (4x4) * (4xN) -> (4xN)
        transformedHomogeneous = (transform @ pointsHomogeneous.T).T
        # Drop homogeneous coordinate and return (N,3)
        return transformedHomogeneous[:, :3]
    
    # -----------------------------------------
    # Convert 6 motion parameters into a 4x4 transformation
    # -----------------------------------------
    @staticmethod
    def smallTransform(parameters):
        """
        Convert 6 motion parameters [rx, ry, rz, tx, ty, tz]
        into a 4x4 homogeneous transformation matrix.
        """
        rX, rY, rZ, tX, tY, tZ = parameters  # unpack rotation + translation

        # Rotation matrices around each axis
        rotationX = np.array([
            [1, 0, 0],
            [0, np.cos(rX), -np.sin(rX)],
            [0, np.sin(rX),  np.cos(rX)]
        ])
        rotationY = np.array([
            [ np.cos(rY), 0, np.sin(rY)],
            [0, 1, 0],
            [-np.sin(rY), 0, np.cos(rY)]
        ])
        rotationZ = np.array([
            [np.cos(rZ), -np.sin(rZ), 0],
            [np.sin(rZ),  np.cos(rZ), 0],
            [0, 0, 1]
        ])

        # Combine rotations (Z-Y-X order)
        combinedRotation = rotationZ @ rotationY @ rotationX

        # Build full 4x4 transformation matrix
        transformationMatrix = np.eye(4)
        transformationMatrix[:3, :3] = combinedRotation
        transformationMatrix[:3, 3] = [tX, tY, tZ]

        return transformationMatrix
    
    # -----------------------------------------
    # Find nearest-neighbor correspondences using a KDTree
    # -----------------------------------------
    def findCorrespondences(self, transformedSource, targetCloud):
        """
        Build a KDTree from the target cloud and find, for each transformed source point,
        the index of its nearest target point.
        """
        kdtree = o3d.geometry.KDTreeFlann(targetCloud)
        correspondences = []
        for p in transformedSource:
            _, idx, _ = kdtree.search_knn_vector_3d(p, 1)
            correspondences.append(idx[0])  #create a correspondences index
        return np.array(correspondences)

    # -----------------------------------------
    # Compute residuals between matched source and target points
    # -----------------------------------------
    def computeResiduals(self, deltaTransformation, sourcePoints, targetPoints, correspondences, currentTransform):
        """
        Compute residuals (differences) between transformed source points
        and their corresponding target points.
        Used by least_squares() as the cost function to minimize.
        """
        newTransformation = deltaTransformation @ currentTransform

        # Transform source points using the combined transformation
        transformedSourcePoints = self.transformPoints(sourcePoints, newTransformation)

        # Compute vector residuals between corresponding points
        residuals = transformedSourcePoints - targetPoints[correspondences]

        # Flatten residuals into 1D vector (required by least_squares)
        return residuals.ravel()
    
    # --- Filter correspondences by distance threshold ---
    def applyDistanceThresholdToCorrespondences(self, sourcePoints, targetPoints, transformedSource, targetCorrespondences):
        # Compute distances between matched points
        matchedTargetPoints = targetPoints[targetCorrespondences]
        differences = transformedSource - matchedTargetPoints
        distances = np.linalg.norm(differences, axis=1)

        # Keep only pairs closer than threshold (e.g., 2 cm)
        inliers = distances < self.distanceThreshold

        # Filter everything by that mask
        filteredSourcePoints = sourcePoints[inliers]
        filteredTargetCorrespondences = targetCorrespondences[inliers]

        fitness = np.sum(inliers) / len(sourcePoints)

        return filteredSourcePoints, filteredTargetCorrespondences, fitness
    
    # -----------------------------------------
    # Define cost function for least-squares optimization
    # -----------------------------------------
    def costFunction(self, parameters, filteredSourcePoints, targetPoints, filteredTargetCorrespondences, currentTransformation):
        #Transform the rx ry... in matriz deltaTransformation
        deltaTransformation = self.smallTransform(parameters)
        # Compute residual result
        residualResult = self.computeResiduals(deltaTransformation, filteredSourcePoints, targetPoints, filteredTargetCorrespondences, currentTransformation)
        return residualResult


    # -----------------------------------------
    # Main ICP optimization loop
    # -----------------------------------------
    def run(self, sourceCloud, targetCloud, globalRegistrationTransformation):
        # Convert point clouds to numpy arrays
        sourcePoints = np.asarray(sourceCloud.points)
        targetPoints = np.asarray(targetCloud.points)

        # Deep copy of initial transformation (from global registration)
        currentTransformation = copy.deepcopy(globalRegistrationTransformation.transformation)

        # ICP iterative optimization
        for i in range(self.maxIterations):
            # Transform source using current estimate
            transformedSource = self.transformPoints(sourcePoints, currentTransformation)

            # Find nearest-neighbor correspondences
            targetCorrespondences = self.findCorrespondences(transformedSource, targetCloud)

            # Apply dinstace threshold
            filteredSourcePoints, filteredTargetCorrespondences, fitness = self.applyDistanceThresholdToCorrespondences(sourcePoints, targetPoints, transformedSource, targetCorrespondences)
            
            #Define the costFunction
            costFunction = partial(
                self.costFunction,
                filteredSourcePoints=filteredSourcePoints,
                targetPoints=targetPoints,
                filteredTargetCorrespondences=filteredTargetCorrespondences,
                currentTransformation=currentTransformation
            )

            # Apply least Squares with cost function and parameters (rx,ry,rz,tx,ty,tz)
            residualResultLeastSquares = least_squares(
                costFunction,
                np.zeros(6),
                verbose=0
            )
                                                    
           
            # Convert result to a 4x4 incremental transformation
            resultLeastSquaresTransformation = self.smallTransform(residualResultLeastSquares.x)  #x vector of optimized parameters

            #print(currentTransformation)
            # -----------------------------------------
            # Update current transformation
            # -----------------------------------------
            currentTransformation = resultLeastSquaresTransformation @ currentTransformation

            #print(currentTransformation)
            # -----------------------------------------
            # Compute mean squared error
            # -----------------------------------------
            rootMeanSquaredError = np.sqrt(np.mean(residualResultLeastSquares.fun ** 2)) # Mean Squared Error (MSE)/ .fun obtain the value of difference between coresponding values
            self.errors.append(rootMeanSquaredError)

            if self.verbose:
                print(f"Iteration {i+1:02d}: error = {rootMeanSquaredError:.6f}")

            # -----------------------------------------
            # Check convergence condition
            # -----------------------------------------
            if np.linalg.norm(residualResultLeastSquares.x) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {i+1}")
                break

        # -----------------------------------------
        # Copy final transformation and compute fitness
        # -----------------------------------------
        self.finalTransform = copy.deepcopy(currentTransformation)
        

        # -----------------------------------------
        # Return the final 4x4 transformation matrix
        # -----------------------------------------
        return self.finalTransform, fitness , rootMeanSquaredError

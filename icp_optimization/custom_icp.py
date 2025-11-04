import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import copy

class CustomICP:
    """Simplified Point-to-Point ICP implementation using SciPy least_squares."""

    def __init__(self, maxIterations=100, tolerance=1e-4, verbose=True):
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
    def smallTransform(params):
        """
        Convert 6 motion parameters [rx, ry, rz, tx, ty, tz]
        into a 4x4 homogeneous transformation matrix.
        Rotations are small angles (radians); translations are in same units as point cloud.
        """
        rx, ry, rz, tx, ty, tz = params  # unpack rotation + translation

        # Rotation matrices around each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx),  np.cos(rx)]
        ])
        Ry = np.array([
            [ np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz),  np.cos(rz), 0],
            [0, 0, 1]
        ])

        # Combine rotations (Z-Y-X order)
        R = Rz @ Ry @ Rx

        # Build full 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]

        return T
    
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
            correspondences.append(idx[0])
        return np.array(correspondences)

    # -----------------------------------------
    # Compute residuals between matched source and target points
    # -----------------------------------------
    def residual(self, params, sourcePoints, targetPoints, correspondences, currentTransform):
        """
        Compute residuals (differences) between transformed source points
        and their corresponding target points.
        Used by least_squares() as the cost function to minimize.
        """
        # 1. Build a small transformation deltaT from params
        deltaT = self.smallTransform(params)

        # 2. Combine deltaT with the current transformation
        T = deltaT @ currentTransform

        # 3. Transform source points using the combined transformation
        transformedSource = self.transformPoints(sourcePoints, T)

        # 4. Compute vector residuals between corresponding points
        residuals = transformedSource - targetPoints[correspondences]

        # 5. Flatten residuals into 1D vector (required by least_squares)
        return residuals.ravel()

    # -----------------------------------------
    # Main ICP optimization loop
    # -----------------------------------------
    def run(self, sourceCloud, targetCloud, globalRegistrationTransformation):
        """
        Runs the custom ICP process to refine alignment between source and target clouds.
        Uses least-squares optimization to minimize point-to-point distance errors.
        """
        # Convert point clouds to numpy arrays
        sourcePoints = np.asarray(sourceCloud.points)
        targetPoints = np.asarray(targetCloud.points)

        # Deep copy of initial transformation (from global registration)
        currentTransform = copy.deepcopy(globalRegistrationTransformation.transformation)

        # ICP iterative optimization
        for i in range(self.maxIterations):
            # -----------------------------------------
            # Step 1: Transform source using current estimate
            # -----------------------------------------
            transformedSource = self.transformPoints(sourcePoints, currentTransform)

            # -----------------------------------------
            # Step 2: Find nearest-neighbor correspondences
            # -----------------------------------------
            correspondences = self.findCorrespondences(transformedSource, targetCloud)

            # --- Filter correspondences by distance threshold ---
            # Compute distances between matched points
            matchedTargetPoints = targetPoints[correspondences]
            diffs = transformedSource - matchedTargetPoints
            distances = np.linalg.norm(diffs, axis=1)

            # Keep only pairs closer than threshold (e.g., 2 cm)
            
            validMask = distances < self.distanceThreshold

            # Filter everything by that mask
            filteredSourcePoints = sourcePoints[validMask]
            filteredCorrespondences = correspondences[validMask]

            # -----------------------------------------
            # Define cost function for least-squares optimization
            # -----------------------------------------
            def costFunc(params):
                return self.residual(params, filteredSourcePoints, targetPoints, filteredCorrespondences, currentTransform)

            # Run nonlinear least-squares optimization to refine small motion
            result = least_squares(costFunc, np.zeros(6), verbose=0)

            # Convert result to a 4x4 incremental transformation
            deltaT = self.smallTransform(result.x)

            # -----------------------------------------
            # Update current transformation
            # -----------------------------------------
            currentTransform = deltaT @ currentTransform

            # -----------------------------------------
            # Compute mean squared error
            # -----------------------------------------
            error = np.mean(result.fun ** 2)
            self.errors.append(error)

            if self.verbose:
                print(f"Iteration {i+1:02d}: error = {error:.6f}")

            # -----------------------------------------
            # Step 6: Check convergence condition
            # -----------------------------------------
            if np.linalg.norm(result.x) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {i+1}")
                break

        # -----------------------------------------
        # Step 7: Copy final transformation and compute fitness
        # -----------------------------------------
        self.finalTransform = copy.deepcopy(currentTransform)

        # -----------------------------------------
        # Step 8: Compute final fitness and RMSE
        # -----------------------------------------
            
        # Transform source with final transformation
        transformedSource = self.transformPoints(sourcePoints, self.finalTransform)

        # Find nearest correspondences one last time
        correspondences = self.findCorrespondences(transformedSource, targetCloud)

        # Compute distances between matched pairs
        matchedTargetPoints = targetPoints[correspondences]
        diffs = transformedSource - matchedTargetPoints
        distances = np.linalg.norm(diffs, axis=1)

        # Determine inliers within threshold
        inliers = distances < self.distanceThreshold
        numInliers = np.sum(inliers)

        # Compute Fitness: fraction of valid (inlier) correspondences
        self.fitness = numInliers / len(sourcePoints) if len(sourcePoints) > 0 else 0.0

        # Compute RMSE over inliers
        if numInliers > 0:
            self.rmse = np.sqrt(np.mean(distances[inliers] ** 2))
        else:
            self.rmse = np.inf  # no valid inliers

        # -----------------------------------------
        # Return the final 4x4 transformation matrix
        # -----------------------------------------
        return self.finalTransform, self.fitness, self.rmse

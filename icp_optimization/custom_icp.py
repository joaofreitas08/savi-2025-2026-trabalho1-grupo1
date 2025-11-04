import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import copy

class CustomICP:
    """Simplified Point-to-Point ICP using SVD (no SciPy, no complexity)."""

    def __init__(self, maxIterations=20, tolerance=1e-6, verbose=True):
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.finalTransform = np.eye(4)
        self.errors = []

    # --- Utility: apply 4x4 matrix to points ---
    @staticmethod
    def transformSourcePoints(points, transform):
        """Apply a 4x4 transformation matrix to Nx3 points."""
        pointsHomogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        pointsInitialTransformed = (transform @ pointsHomogeneous.T).T[:, :3]
        return pointsInitialTransformed

    def run(self, sourceCloud, targetCloud):
        sourcePoints = np.asarray(sourceCloud.points)
        targetPoints = np.asarray(targetCloud.points)
        currentTransform = np.eye(4)

        kdtree = o3d.geometry.KDTreeFlann(targetCloud)

        for i in range(self.maxIterations):
            # --- 1. Find correspondences ---
            matchedPoints = []
            for p in sourcePoints:
                _, idx, _ = kdtree.search_knn_vector_3d(p, 1)
                matchedPoints.append(targetPoints[idx[0]])
            matchedPoints = np.array(matchedPoints)

            # --- 2. Compute best rotation and translation using SVD ---
            mu_src = np.mean(sourcePoints, axis=0)
            mu_tgt = np.mean(matchedPoints, axis=0)

            src_centered = sourcePoints - mu_src
            tgt_centered = matchedPoints - mu_tgt

            H = src_centered.T @ tgt_centered
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T
            t = mu_tgt - R @ mu_src

            # --- 3. Update transform and apply it ---
            deltaTransform = np.eye(4)
            deltaTransform[:3, :3] = R
            deltaTransform[:3, 3] = t
            currentTransform = deltaTransform @ currentTransform
            sourcePoints = (R @ sourcePoints.T).T + t

            # --- 4. Compute mean error ---
            error = np.mean(np.linalg.norm(sourcePoints - matchedPoints, axis=1))
            self.errors.append(error)
            if self.verbose:
                print(f"Iteration {i+1:02d}: mean error = {error:.6f}")

            # --- 5. Stop if small change ---
            if i > 0 and abs(self.errors[-2] - self.errors[-1]) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {i+1}")
                break

        self.finalTransform = currentTransform
        return currentTransform
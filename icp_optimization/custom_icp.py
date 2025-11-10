import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import copy
from functools import partial
from scipy.spatial.transform import Rotation as R

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
        self.voxelSize = 0.05
        self.distanceThreshold = self.voxelSize * 10   #1.5            

    # -----------------------------------------
    # Apply a 4x4 transformation matrix to Nx3 points
    # -----------------------------------------
    @staticmethod
    def transformPoints(points, transform):
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
    def findCorrespondences(self, sourceCloud, targetCloud):
        # Build kdtree from target cloud
        kdtree = o3d.geometry.KDTreeFlann(targetCloud)

        #Create a list of correspondences
        sourceToTargetCorrespondencesIndex = []

        for point in sourceCloud:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1) # 1 = only closer point
            sourceToTargetCorrespondencesIndex.append(idx[0])  #create a correspondences index (0). what point its the closest

        return np.array(sourceToTargetCorrespondencesIndex)
    

    # -----------------------------------------
    # Filter correspondences by distance threshold
    # -----------------------------------------
    def applyDistanceThresholdToCorrespondences(self, distances):
        # Compute distances between matched points
        inliers = []
        for distance in distances: #pointClouds in what
        # Keep only pairs closer than threshold 
            if distance < self.distanceThreshold:
                inliers.append([1])
            else:
                inliers.append([0])

        inliersNp = np.array(inliers)
        #print(inliersNp)
        
        #Transform in np array
        filteredDistances = []
        for distance, inlier in zip(distances, inliersNp):
            # Filter everything by that mask
            filteredDistances.append(distance * inlier)

           #fitness = np.sum(inliers) / len(sourcePoints)
        return np.array(filteredDistances)
    
    def getFromTransformedSourceCloudParameters(self, transformedSourceCloud):

        rotationMatrix = transformedSourceCloud[:3, :3]
        translationVector = transformedSourceCloud[:3, 3]

        rotation = R.from_matrix(rotationMatrix)
        rx, ry, rz = rotation.as_euler('xyz', degrees=False)

        parameters = np.hstack([rx, ry, rz, translationVector])

        return parameters
    

    # -----------------------------------------
    # Define cost function for least-squares optimization
    # -----------------------------------------
    def costFunction(self, parameters, filteredSourcePoints, filteredMatchedTargetPoints):
        #Transform the rx ry... in matriz deltaTransformation
        deltaTransformation = self.smallTransform(parameters)
        print(deltaTransformation)
        #print(deltaTransformation)
        # Compute residual result
        residualResult = self.computeResiduals(deltaTransformation, filteredSourcePoints, filteredMatchedTargetPoints)
        return residualResult

    # -----------------------------------------
    # Compute residuals between matched source and target points
    # -----------------------------------------
    def computeResiduals(self, deltaTransformation, filteredSourcePoints, filteredMatchedTargetPoints):


        # Transform source points using the combined transformation
        transformedSourcePoints = self.transformPoints(filteredSourcePoints, deltaTransformation)

        # Compute vector residuals between corresponding points
        residuals = transformedSourcePoints - filteredMatchedTargetPoints

        # Flatten residuals into 1D vector (required by least_squares)
        return residuals.ravel()
    

    # -----------------------------------------
    # Main ICP optimization loop
    # -----------------------------------------
    def run(self, sourceCloud, targetCloud, globalRegistrationTransformation):

        sourcePoints = np.asarray(sourceCloud.points)
        targetPoints = np.asarray(targetCloud.points)
        # Deep copy of initial transformation (from global registration)
        currentTransformation = copy.deepcopy(globalRegistrationTransformation.transformation)
        
        
        # ICP iterative optimization
        for i in range(self.maxIterations):
            # Transform source using current estimate
            transformedSourceCloud = self.transformPoints(sourcePoints, currentTransformation) #Nx3


            parameters = self.getFromTransformedSourceCloudParameters(currentTransformation)

            


            # Find nearest-neighbor correspondences
            sourceToTargetCorrespondencesIndex = self.findCorrespondences(transformedSourceCloud, targetCloud)

            #print(sourceToTargetCorrespondencesIndex)

            #Calculate distances
            matchedTargetPoints = []
            for idx in sourceToTargetCorrespondencesIndex:
                matchedTargetPoints.append(targetPoints[idx])

            #print(matchedTargetPoints)

            matchedTargetPoints = np.array(matchedTargetPoints)
            differences = transformedSourceCloud - matchedTargetPoints
            distances = np.linalg.norm(differences, axis=1)

            

            # Apply distancece threshold
            filteredDistances = self.applyDistanceThresholdToCorrespondences(distances)
            #print (filteredDistances)
            
            filteredMatchedTargetPoints = matchedTargetPoints * filteredDistances
            filteredSourcePoints = transformedSourceCloud * filteredDistances
           
            #print(filteredMatchedTargetPoints)

            #filteredMatchedTargetPointsHomogeneous = np.hstack([filteredMatchedTargetPoints, np.ones((filteredMatchedTargetPoints.shape[0], 1))])

            #print(filteredMatchedTargetPointsHomogeneous)



            #Define the costFunction
            costFunction = partial(
                self.costFunction,
                filteredSourcePoints=filteredSourcePoints,
                filteredMatchedTargetPoints=filteredMatchedTargetPoints,
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


            print(currentTransformation)
            #print(np.linalg.det(currentTransformation[:3, :3]))

            #print(currentTransformation)
            # -----------------------------------------
            # Compute mean squared error
            # -----------------------------------------
            rootMeanSquaredError = np.sqrt(np.mean(residualResultLeastSquares.fun ** 2)) # Mean Squared Error (MSE)/ .fun obtain the value of difference between coresponding values

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
        return self.finalTransform , rootMeanSquaredError

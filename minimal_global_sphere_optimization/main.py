from scipy.optimize import minimize
import numpy as np
import open3d as o3d

def sphereObjective(params, points):
    return params[3]   # minimizar o raio

def constraint_factory(points):
    def c(params):
        xc, yc, zc, r = params
        center = np.array([xc, yc, zc])
        distances = np.linalg.norm(points - center, axis=1)
        return r - distances  # >= 0
    return c

def main():
    pointCloud = o3d.io.read_point_cloud("pcd_to_work/cloud_registered.pcd")
    points = np.asarray(pointCloud.points)

    # first guess
    center0 = points.mean(axis=0)
    radius0 = np.max(np.linalg.norm(points - center0, axis=1))

    x0 = np.array([center0[0], center0[1], center0[2], radius0])

    constraints = ({
        'type': 'ineq',
        'fun': constraint_factory(points)
    })

    result = minimize(
        sphereObjective,
        x0,
        args=(points,),
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 200, 'ftol': 1e-12}
    )

    xc, yc, zc, r = result.x
    center = np.array([xc, yc, zc])
    radius = abs(r)

    print("Centro otimizado:", center)
    print("Raio otimizado:", radius)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.translate(center)
    sphere.compute_vertex_normals()

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [1.0, 0.0, 0.0, 0.3]

    o3d.visualization.draw([
        {"name": "cloud",  "geometry": pointCloud},
        {"name": "sphere", "geometry": sphere, "material": material}
    ])

if __name__ == '__main__':
    main()

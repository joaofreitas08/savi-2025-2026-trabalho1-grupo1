#!/usr/bin/env python3
import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from functools import partial
import time


# Melhoria na Função Objetivo
def sphereObjective(params, points):
    # params = [xc, yc, zc, r]
    r = params[3]
    return r**2 # Minimizar o raio ao quadrado

#minimize r knowing that all the points should be inside th sphere


# Melhoria na Função de Restrição (r² - d² >= 0)
def sphereConstraint(params, points):
    xc, yc, zc, r = params
    center = np.array([xc, yc, zc])

    # 1. Distâncias Quadradas
    # np.sum(..., axis=1) calcula a soma dos quadrados das diferenças (a distância euclidiana ao quadrado)
    distancesSquare = np.sum((points - center)**2, axis=1)
    
    # 2. Restrição Otimizada (r² - dist² >= 0)
    return r**2 - distancesSquare


# -----------------------------------------
# Inicializar visualizador antigo Open3D
# -----------------------------------------
def initVisualizer(points):
    vis = o3d.visualization.Visualizer()
    vis.create_window("Sphere Optimization Animation", 1024, 768)

    # Obter opções de renderização PRIMEIRO
    renderOpt = vis.get_render_option()

    # 1. Geometria da cloud (Pontos Sólidos)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.paint_uniform_color([0.8, 0.8, 0.8])
    vis.add_geometry(cloud)

    # 2. Esfera será substituída a cada iteração
    sphere = o3d.geometry.TriangleMesh()
    vis.add_geometry(sphere)

    # 3. Configurações de Renderização CRUCIAIS para Transparência da Malha
    renderOpt.mesh_show_back_face = True # Vê o interior da malha
    renderOpt.mesh_color_option = o3d.visualization.MeshColorOption.Color # Usa a cor definida

    # 4. Outras opções
    renderOpt.point_size = 2.0

    vis.poll_events()
    vis.update_renderer()

    return vis, sphere


def iterationCallback(params, points, vis, sphere):
    xc, yc, zc, r = params
    r = abs(r)
    center = np.array([xc, yc, zc])

    tmp = o3d.geometry.TriangleMesh.create_sphere(r, resolution=15)
    tmp.compute_vertex_normals()
    tmp.translate(center)

    # COR VISÍVEL (SEM TRANSPARÊNCIA)
    color = np.array([0.7, 0.8, 1.0])
    colors = np.tile(color, (np.asarray(tmp.vertices).shape[0], 1))
    tmp.vertex_colors = o3d.utility.Vector3dVector(colors)

    sphere.vertices = tmp.vertices
    sphere.triangles = tmp.triangles
    sphere.vertex_normals = tmp.vertex_normals
    sphere.triangle_normals = tmp.triangle_normals
    sphere.vertex_colors = tmp.vertex_colors

    vis.update_geometry(sphere)
    vis.poll_events()
    vis.update_renderer()

    time.sleep(1)


# -----------------------------------------
# MAIN
# -----------------------------------------
def main():

    # -----------------------------------------
    # Load da point cloud
    # -----------------------------------------
    cloud = o3d.io.read_point_cloud("pcd_to_work/cloud_registered.pcd")
    points = np.asarray(cloud.points)

    # -----------------------------------------
    # First guess: centro e raio inicial
    # -----------------------------------------
    initialCenter = points.mean(axis=0)
    initialRadius = np.max(np.linalg.norm(points - initialCenter, axis=1))

    x0 = np.array([initialCenter[0],
                   initialCenter[1],
                   initialCenter[2],
                   initialRadius])

    # -----------------------------------------
    # Visualizador para animação
    # -----------------------------------------
    vis, sphere = initVisualizer(points)

    # -----------------------------------------
    # Constraints com partial
    # -----------------------------------------
    cons = ({
        "type": "ineq",
        "fun": partial(sphereConstraint, points=points)
    })

    # -----------------------------------------
    # Objective + callback com partial
    # -----------------------------------------
    wrappedCallback = partial(iterationCallback,
                              points=points,
                              vis=vis,
                              sphere=sphere)

    wrappedObjective = partial(sphereObjective,
                               points=points)

    # -----------------------------------------
    # Minimização com animação
    # -----------------------------------------
    result = minimize(
        fun=wrappedObjective,
        x0=x0,
        method="SLSQP",
        constraints=cons,
        callback=wrappedCallback,
        options={"maxiter": 200, "ftol": 1e-12, "disp": True}
    )

    # Fechar janela de animação
    vis.destroy_window()

    # -----------------------------------------
    # Resultados finais
    # -----------------------------------------
    xc, yc, zc, r = result.x
    center = np.array([xc, yc, zc])
    radius = abs(r)

    print("\nCentro otimizado:", center)
    print("Raio otimizado:", radius)

    # -----------------------------------------
    # Visualização final transparente REAL
    # -----------------------------------------
    sphereFinal = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphereFinal.translate(center)
    sphereFinal.compute_vertex_normals()

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [1, 0, 0, 0.25]

    o3d.visualization.draw([
        {"name": "cloud",  "geometry": cloud},
        {"name": "sphere", "geometry": sphereFinal, "material": material}
    ])


# -----------------------------------------
# Entry point
# -----------------------------------------
if __name__ == "__main__":
    main()

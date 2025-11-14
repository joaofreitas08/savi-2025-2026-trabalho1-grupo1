# Practical Assignment 1 — SAVI  
![Python](https://img.shields.io/badge/python-3.10+-blue)  
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## Overview

This assignment is divided into **four tasks**, each addressing a different stage of a 3D perception and processing pipeline:

### 1. image_to_cloud
Converts an RGB image and its corresponding depth image into a 3D point cloud.

### 2. icp_tutorial  
Uses Open3D’s built-in ICP (Iterative Closest Point) to align multiple point clouds.

### 3. icp_optimization  
Implements a **custom ICP algorithm** using `scipy.optimize.least_squares`.  
This includes:
- building your own residual function  
- computing point correspondences  
- visualizing the ICP evolution over iterations  

### 4. minimal_global_sphere_optimization  
Computes the **minimum enclosing sphere** of a point cloud using nonlinear optimization (SLSQP), with real-time visualization in the Open3D GUI.

Each module demonstrates a different technique used in 3D perception pipelines.



---
## How to Install

1. **Clone the repository**
   ```bash
   git clone https://github.com/joaofreitas08/savi-2025-2026-trabalho1-grupo1.git
    ```

2. **Install python dependencies**
    ```bash
    sudo pip install -r requirements.txt
    ```  
---
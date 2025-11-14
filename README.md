# Practical Assignment 1 — SAVI  
![Python](https://img.shields.io/badge/python-3.10+-blue)  
![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## Overview

This assignment is divided into **four tasks**, each addressing a different stage of a 3D perception and processing pipeline:

### 1. Image (depth and RGB) to point cloud
Converts an RGB image and its corresponding depth image into a 3D point cloud.
For more details, check [these example](docs/examples).

### 2. ICP tutotial from Open3D 
Uses Open3D’s built-in ICP (Iterative Closest Point) to align multiple point clouds.
For more details, check [these example](docs/examples).

### 3. ICP custom optimization
Implements a **custom ICP algorithm** using `scipy.optimize.least_squares`.  
This includes:
- building own residual function  
- computing point correspondences  
- visualizing the ICP evolution over iterations  

For more details, check [these example](docs/examples).

### 4. Minimal global sphere oprimization 
Computes the **minimum enclosing sphere** of a point cloud using nonlinear optimization (SLSQP), with real-time visualization in the Open3D GUI.
For more details, check [these example](docs/examples).



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
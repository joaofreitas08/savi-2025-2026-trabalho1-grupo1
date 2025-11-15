# Point Cloud Registration (Global + Custom ICP)

This project performs **incremental registration of 3D point clouds** using:

- **Global Registration (RANSAC)**
- **Custom ICP refinement** implemented from scratch using `scipy.optimize.least_squares`
- A **real-time Open3D visualizer** showing each ICP iteration
- An **interactive Open3D GUI** for toggling visibility and modifying colors of:
  - Original point clouds  
  - Globally registered clouds  
  - Fully registered clouds  

The goal is to provide a **transparent, debuggable, and visually interpretable registration pipeline**, ideal for learning, experimentation, and analysis.

---

## Project Highlights

### Custom ICP (Main Contribution)

The heart of this project is a **fully custom ICP implementation**, featuring:

- Point-to-point ICP  
- KD-Tree correspondence search  
- Nonlinear optimization using SciPy's `least_squares` with Huber loss  
- Gradual 3D pose updates (small rotation + translation steps)
- Real-time visualization callback per iteration  


---

## How to Run

```bash
python3 icp_optimization/main.py
```

---

## What you should expect
Target cloud (fixed) → green

Source cloud (moving) → red

<td><img src="images\gif_icp_opt.gif" width="1000"/></td>

Below is an example showing the original clouds, global alignment, and final ICP refinement:

<table>
  <tr>
    <th>Without Registration</th>
    <th>Global Registration</th>
    <th>ICP Registration</th>
  </tr>
  <tr>
    <td><img src="images\2normalimages.png" width="300"/></td>
    <td><img src="images\imagewithglobalregist.png" width="300"/></td>
    <td><img src="images\imageregistered.png" width="300"/></td>
  </tr>
</table>




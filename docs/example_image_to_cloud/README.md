# RGB-D to Point Cloud Conversion
This module demonstrates how to convert RGB and Depth images from the TUM RGB-D Dataset into 3D point clouds using Open3D.
It also includes an interactive GUI where you can toggle visibility and change the color of each generated point cloud.


---

### How to run the code
To run the script with the default dataset folders:

```bash
python3 image_to_cloud/main.py
```

If you want to change the images (RGB or depth) go to:
```bash
image_to_cloud/tum_dataset/rgb
image_to_cloud/tum_dataset/depth
```

If you want to access the point clouds go to:
```bash
image_to_cloud/output_cloud
```

### Input Example (RGB + Depth)

<table>
  <tr>
    <th>RGB Image 1</th>
    <th>Depth Image 1</th>
  </tr>
  <tr>
    <td><img src="images\tum_dataset\rgb\1.png" width="450"/></td>
    <td><img src="images\tum_dataset\depth\1.png" width="450"/></td>
  </tr>
  <tr>
    <th>RGB Image 2</th>
    <th>Depth Image 2</th>
  </tr>
  <tr>
    <td><img src="images\tum_dataset\rgb\2.png" width="450"/></td>
    <td><img src="images\tum_dataset\depth\2.png" width="450"/></td>
  </tr>
</table>

---

### Output Example (Point Clouds)

<table>
  <tr>
    <th>Point Cloud 1</th>
    <th>Point Cloud 2</th>
  </tr>
  <tr>
    <td><img src="images\pointCLouds\pointcloud1.png" width="450"/></td>
    <td><img src="images\pointCLouds\pointcloud2.png" width="450"/></td>
  </tr>
</table>




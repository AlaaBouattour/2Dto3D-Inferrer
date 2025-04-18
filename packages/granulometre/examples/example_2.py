# examples/example_2.py

"""
Example 2: 3D Visualization of a Rotated Ellipsoid and its 2D Projection.

This example demonstrates how to:
  1. Generate a 3D ellipsoid (axes a, b, c) using a Uniform distribution.
  2. Apply a random rotation to the ellipsoid.
  3. Visualize the rotated ellipsoid in 3D.
  4. Compute and overlay its 2D projection (as an ellipse on the XY-plane) using the same rotation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d

import granulometre
from utils import create_ellipsoid  

def main():
    # Ellipsoid parameters
    a, b, c = granulometre.distribution.Uniform([2, 8]).sample()  # generates axes in [2, 10]^3
    center = [0, 0, 0]
    
    # Generate a random rotation matrix using the QR method
    rot_gen = granulometre.rotation.Generator(method='qr')
    R = rot_gen.random_rotation_matrix()
    
    # Calculate the 2D projection of the ellipsoid using the same rotation
    a_proj, b_proj, angle = granulometre.ellipsoid_projection_axes(a, b, c, R=R, method='qr')
    proj_angle_deg = np.degrees(angle)
    
    print(f"Ellipsoid semi-axes: {a}, {b}, {c}")
    print(f"Projected ellipse semi-axes: {a_proj:.2f}, {b_proj:.2f}")
    print(f"Orientation angle: {proj_angle_deg:.2f} degrees")
    
    # Create the 3D ellipsoid mesh with rotation applied.
    x, y, z = create_ellipsoid(a, b, c, R, center)
    
    # Setup the figure with two subplots.
    fig = plt.figure(figsize=(12, 8))
    
    # 3D subplot for ellipsoid and its projection.
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot the rotated ellipsoid.
    ax1.plot_surface(x, y, z, color='b', alpha=0.3, rstride=4, cstride=4)
    
    # Generate points for the projection ellipse on the XY plane.
    u = np.linspace(0, 2*np.pi, 100)
    projection_x = a_proj * np.cos(u) * np.cos(angle) - b_proj * np.sin(u) * np.sin(angle)
    projection_y = a_proj * np.cos(u) * np.sin(angle) + b_proj * np.sin(u) * np.cos(angle)
    projection_z = np.zeros_like(u)
    
    # Draw the projection ellipse on the XY plane.
    ax1.plot(projection_x, projection_y, projection_z, 'r-', linewidth=3)
    
    # Create a semi-transparent green XY plane.
    xx, yy = np.meshgrid(np.linspace(-6, 6, 10), np.linspace(-6, 6, 10))
    zz = np.zeros_like(xx)
    ax1.plot_surface(xx, yy, zz, alpha=0.1, color='g')
    
    # Draw coordinate axes.
    axis_length = max(a, b, c) * 1.5
    ax1.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1)
    ax1.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1)
    ax1.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1)
    
    # Set labels, limits, and title for the 3D plot.
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-axis_length, axis_length])
    ax1.set_ylim([-axis_length, axis_length])
    ax1.set_zlim([-axis_length, axis_length])
    
    # Annotate the 3D plot with the ellipsoid and projection parameters.
    ax1.text2D(0.05, 0.95, f"3D Axes: a={a:.2f}, b={b:.2f}, c={c:.2f}", transform=ax1.transAxes, fontsize=10, color='black')
    ax1.text2D(0.05, 0.90, f"Proj: A={a_proj:.2f}, B={b_proj:.2f}, Angle={proj_angle_deg:.2f}°", transform=ax1.transAxes, fontsize=10, color='red')
    ax1.set_title('3D Ellipsoid and its XY Projection')
    
    
    
    # 2D subplot for the projection.
    ax2 = fig.add_subplot(122)
    
    # Create an ellipse patch for the projection.
    ellipse = patches.Ellipse((0, 0), 2*a_proj, 2*b_proj, 
                              angle=proj_angle_deg, 
                              edgecolor='r', facecolor='none', linewidth=2)
    ax2.add_patch(ellipse)
    
    # Draw principal axes for clarity.
    ax2.plot([0, a_proj*np.cos(angle)], [0, a_proj*np.sin(angle)], 'r--', linewidth=1)
    ax2.plot([0, -b_proj*np.sin(angle)], [0, b_proj*np.cos(angle)], 'r--', linewidth=1)
    
    # Plot points along the ellipse contour.
    ax2.plot(projection_x, projection_y, 'r.', markersize=1)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim([-axis_length, axis_length])
    ax2.set_ylim([-axis_length, axis_length])
    ax2.set_title('2D Projection of the Ellipsoid')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

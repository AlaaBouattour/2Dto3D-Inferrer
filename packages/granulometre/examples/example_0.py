# examples/example_0.py
"""
Example 0: Basic Usage of the Granulometre Package.

This example demonstrates how to:
  1. Create a Uniform distribution for 3D ellipsoids (parameterized by [u, l], meaning the ellipsoids are generated in [u, u+l]^3).
  2. Instantiate a Simulator with that distribution.
  3. Generate a sample of ellipsoids and obtain their 2D projection axes.
  4. Compute the 2D projection parameters (semi-axes and orientation angle) using a random rotation.
"""

import granulometre

# Create a Uniform distribution for 3D ellipsoids.
# The distribution is parameterized by [u, l] meaning ellipsoids are generated in [u, u+l]^3.
dist = granulometre.distribution.Uniform([5.0, 5.0])

# Create a Simulator with the given distribution.
sim = granulometre.Simulator(distribution=dist)

# Generate a sample of ellipsoids and get the 2D projection axes.
samples = sim.generate_samples(10)  # Returns an array of shape (10, 2)
print("Projected ellipse axes (A, B):")
print(samples)

# Generate a random rotation matrix using the Generator class.

rot_gen = granulometre.rotation.Generator(method='qr')
R = rot_gen.random_rotation_matrix()

# Compute the 2D projection parameters using the same rotation.

A_proj, B_proj, proj_angle = granulometre.ellipsoid_projection_axes(5.0, 4.0, 3.0, R=R, method='qr')
print(f"2D Projection: A_proj = {A_proj}, B_proj = {B_proj}, Angle (radians) = {proj_angle}")

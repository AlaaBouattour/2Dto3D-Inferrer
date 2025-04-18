# Granulometre

**Granulometre** is a Python package designed to simulate the granulometer process by generating 3D ellipsoids that represent particle shapes, applying random rotations, and computing their 2D projections. The package supports various parametric distributions for the ellipsoids and provides dedicated modules for generation, rotation, and projection.

## Features

- **3D Ellipsoid Generation:**  
  Generate ellipsoids with semi-axes (a, b, c) using various parametric distributions (e.g., Uniform, Beta).

- **Rotation:**  
  Apply random rotations to the ellipsoids using the `Generator` class in the `rotation` submodule. For example:
  ```python
  from granulometre.rotation import Generator
  R = Generator(method='qr').random_rotation_matrix()
  ```

- **Projection:**  
  Compute the 2D projection of a rotated ellipsoid onto the XY-plane. The projection function returns the semi-axes of the resulting ellipse and its orientation angle (in radians).

- **Simulation:**  
  The `Simulator` class integrates ellipsoid generation, rotation, and projection, providing an end-to-end simulation of the granulometer process.

## Installation

This package uses a `pyproject.toml` for build configuration. To install the package in editable mode:

1. Navigate to the package directory (the folder containing `pyproject.toml`):
   ```bash
   cd biomasse/packages/granulometre
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

Below is a minimal usage example:

```python
import granulometre

# Create a Uniform distribution for 3D ellipsoids.
# The distribution is parameterized by [u, l] such that ellipsoids are generated in [u, u+l]^3.
dist = granulometre.distribution.Uniform([5.0, 5.0])

# Create a Simulator using that distribution.
sim = granulometre.Simulator(distribution=dist)

# Generate a sample of ellipsoids and obtain the projected ellipse axes.
samples = sim.generate_samples(10)  # Returns an array of shape (10, 2)
print("Projected ellipse axes (A, B):", samples)

# Generate a random rotation matrix using the Generator class.
from granulometre.rotation import Generator
rot_gen = Generator(method='qr')
R = rot_gen.random_rotation_matrix()

# Compute the 2D projection parameters using the same rotation.
from granulometre.projection import projection
A_proj, B_proj, proj_angle = projection.ellipsoid_projection_axes(5.0, 4.0, 3.0, R=R, method='qr')
print(f"2D Projection: A_proj = {A_proj}, B_proj = {B_proj}, Angle (radians) = {proj_angle}")
```

## Project Structure

The package is organized as follows:

```
granulometre/
├── pyproject.toml            # Build configuration for the granulometre package
├── README.md                # This file
└── src/
    └── granulometre/
        ├── __init__.py      # Exposes main classes and modules
        ├── distribution/    # Contains modules defining various distributions (e.g., Uniform, Beta, etc.)
        │   └── ...          
        ├── generator/       # Contains simulation logic, including the Simulator class
        │   ├── __init__.py  
        │   ├── generator.py
        │   └── distributions.py
        ├── rotation/        # Contains rotation functions/classes
        │   ├── __init__.py  
        │   └── random_rotation.py   # Contains class "Generator" for generating random rotation matrices
        └── projection/      # Contains functions to compute 2D projections
            ├── __init__.py  
            └── projection.py
```

## Examples

The package includes an `examples/` folder (located within the package repository) with progressive usage examples:

- **example_0.py:** Basic usage showing sample generation and projection parameter computation.
- **example_1.py:** Histogram analysis of generated ellipses.
- **example_2.py:** 3D visualization of a rotated ellipsoid and its 2D projection.
  
Each example is designed to demonstrate specific functionalities of the package.

## Contributing

## License

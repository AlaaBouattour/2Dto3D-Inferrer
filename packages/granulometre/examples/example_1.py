# examples/example_1.py

"""
Example 1: Histogram Analysis of Generated Ellipses.

This example demonstrates how to:
  1. Generate a large number of ellipsoids using the Simulator.
  2. Analyze the distribution of the projected ellipse axes (A, B) and the rotation angles.
  3. Plot histograms for these characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
import granulometre

def main():
    # 1) Create a Uniform distribution for 3D ellipsoids.
    #    The distribution is parameterized by [u, l] so that ellipsoids are generated in [u, u+l]^3.
    u_value = 4.0
    l_value = 3.0  # ellipsoids in [4, 7]^3
    dist = granulometre.distribution.Uniform([u_value, l_value])
    
    # 2) Create a Simulator with the given distribution.
    sim = granulometre.Simulator(distribution=dist)
    
    # 3) Generate a large sample of ellipsoids.
    #    The generate_samples method returns a 2D array of shape (N, 2) for the projected ellipse axes,
    #    and the rotation angles (in radians) are stored in sim.angles.
    N = 10000
    samples = sim.generate_samples(N)
    A_vals = samples[:, 0]
    B_vals = samples[:, 1]
    angles = np.array(sim.angles)
    angles_deg = np.degrees(angles) % 180  # Normalize angles to [0, 180)
    
    # Print summary statistics.
    print(f"Generated {N} samples.")
    print(f"Mean A: {np.mean(A_vals):.2f}, Std A: {np.std(A_vals):.2f}")
    print(f"Mean B: {np.mean(B_vals):.2f}, Std B: {np.std(B_vals):.2f}")
    print(f"Mean angle: {np.mean(angles_deg):.2f}°, Std angle: {np.std(angles_deg):.2f}°")
    
    # 4) Plot histograms for A, B and rotation angle.
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].hist(A_vals, bins=40, color='blue', alpha=0.7)
    axs[0].set_title("Histogram of A (Major Axis)")
    axs[0].set_xlabel("A")
    axs[0].set_ylabel("Frequency")
    
    axs[1].hist(B_vals, bins=40, color='green', alpha=0.7)
    axs[1].set_title("Histogram of B (Minor Axis)")
    axs[1].set_xlabel("B")
    axs[1].set_ylabel("Frequency")
    
    axs[2].hist(angles_deg, bins=40, color='red', alpha=0.7)
    axs[2].set_title("Histogram of Rotation Angles")
    axs[2].set_xlabel("Angle (°)")
    axs[2].set_ylabel("Frequency")
    
    # annotations
    distribution_text = f"Distribution: Uniform on [u, u+l]^3 with u = {u_value} and l = {l_value}"
    fig.text(0.5, 0.96, distribution_text, ha='center', fontsize=12, color='purple')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])  
    plt.show()

if __name__ == "__main__":
    main()
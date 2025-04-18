# examples/example_1.py

"""
Example 1: Integrated ABC Sampling Example using the Granulometre Package.

WARNING: This example requires that the "granulometre" package is installed.
It uses granulometre to generate synthetic 3D ellipsoid data and then applies ABC
rejection sampling (from the abc_sampling package) to infer the underlying parameters.

This example demonstrates how to:
  1. Generate synthetic observed data using the granulometre package.
  2. Define a prior for the simulator parameters.
  3. Run the ABC Rejection Sampling to infer the parameters.
  4. Plot histograms of the accepted parameters and an interpolated parameter-to-distance map.
"""

import numpy as np
import matplotlib.pyplot as plt
import abc_sampling

# Ensure granulometre is available.
try:
    import granulometre
except ImportError:
    raise ImportError("The 'granulometre' package is required to run this example. Please install it and try again.")

from sklearn.svm import SVR

def main():
    # True parameters for the granulometre distribution:
    # Here, the Uniform distribution is parameterized by [u, l] (ellipsoids in [u, u+l]^3)
    u_true = 3.0
    l_true = 4.0  # True ellipsoids are generated in [3, 7]^3.
    
    # Generate observed data using the granulometre package.
    true_dist = granulometre.distribution.Uniform([u_true, l_true])
    sim = granulometre.Simulator(distribution=true_dist)
    N = 1000
    observed_data = sim.generate_samples(N)
    
    # Define a prior for the simulator parameters.
    # In this example, the prior is uniform on [0, 10] for the first parameter (u)
    # and on [0, 10] for the second parameter (l).
    prior = abc_sampling.priors.Uniform([0, 10, 0, 10])
    
    # Use a distance function (here, the Wasserstein distance is used in abc_sampling.distance_function).
    distance_func = abc_sampling.distance_function
    
    # Create an ABC Rejection Sampler using the granulometre Simulator.
    abc_sampler = abc_sampling.RejectionSampling(
        simulator=sim,
        dataAB=observed_data,
        prior_sampler=prior,
        distance_func=distance_func
    )
    
    # Run the ABC rejection sampling.
    abc_sampler.run(N_iter=1000, epsilon=0.5)
    
    accepted_params = np.array(abc_sampler.get_accepted_params())
    history = abc_sampler.get_history()
    
    print("Accepted parameters:")
    if len(accepted_params) < 100:
        print(accepted_params)
    else:
        print(len(accepted_params), " parameters accepted.")
    
    # Plot histograms of the accepted parameters along with true parameter values.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    axs[0].hist(accepted_params[:, 0], bins=20, color='blue', alpha=0.7)
    axs[0].axvline(u_true, color='red', linestyle='dashed', linewidth=2, label='True u')
    axs[0].set_title("Histogram of u (Parameter 1)")
    axs[0].set_xlabel("u")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()
    
    axs[1].hist(accepted_params[:, 1], bins=20, color='green', alpha=0.7)
    axs[1].axvline(l_true, color='red', linestyle='dashed', linewidth=2, label='True l')
    axs[1].set_title("Histogram of l (Parameter 2)")
    axs[1].set_xlabel("l")
    axs[1].set_ylabel("Frequency")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    #fit an interpolator to the history and plot the parameter-to-distance surface.
    regressor = SVR(kernel='rbf', C=1, gamma='scale')
    interpolator = abc_sampling.Interpolator(history=history, regressor=regressor)
    interpolator.fit()
    
    plotter = abc_sampling.HistoryPlotter(history=history, model=interpolator)
    plotter.plot_interpolated_surface(grid_size=100)

if __name__ == "__main__":
    main()

# examples/example_0.py

"""
Toy Example for ABC Sampling using a Gaussian Simulator

This example demonstrates how to:
  1. Generate synthetic "observed" data from a Gaussian distribution using GaussianSimulator.
  2. Define a prior for the simulator parameters.
  3. Use the ABC Rejection Sampling method to infer the parameters.
  4. Plot histograms of the accepted parameters along with the true parameter values.
"""

import numpy as np
import matplotlib.pyplot as plt
import abc_sampling
from sklearn.svm import SVR
from utils import GaussianSimulator

def main():
    # True parameters for the Gaussian distribution.
    true_params = [7.0, 1.5]  # mean = 7.0 and std = 1.5
    simulator = GaussianSimulator(true_params)
    observed_data = simulator.generate_samples(1000)
    
    # Define a prior for the simulator parameters.
    # For example, we expect the mean to be between 0 and 10, and std between 0 and 5.
    # The prior sampler is expected to return a 1D array, [mean, std]
    prior = abc_sampling.priors.Uniform([0, 10, 0, 5])
    
    # Use a distance function (here, the Wasserstein distance)
    distance_func = abc_sampling.distance_function
    
    # Create an ABC Rejection Sampler with our GaussianSimulator.
    abc_sampler = abc_sampling.RejectionSampling(
        simulator=simulator,
        dataAB=observed_data,
        prior_sampler=prior, 
        distance_func=distance_func
    )
    
    # Run the ABC rejection sampling.
    abc_sampler.run(N_iter=10000, epsilon=0.5)
    
    accepted_params = np.array(abc_sampler.get_accepted_params())
    history = abc_sampler.get_history()
    
    print("Accepted parameters:")
    if len(accepted_params) < 100:
        print(accepted_params)
    else:
        print(len(accepted_params), " parameters")
    
    # Plot histograms of the accepted parameters along with the true values.
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(accepted_params[:, 0], bins=20, color='blue', alpha=0.7)
    plt.axvline(true_params[0], color='red', linestyle='dashed', linewidth=2, label='True Mean')
    plt.title("Histogram of Mean")
    plt.xlabel("Mean")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(accepted_params[:, 1], bins=20, color='green', alpha=0.7)
    plt.axvline(true_params[1], color='red', linestyle='dashed', linewidth=2, label='True Std')
    plt.title("Histogram of Standard Deviation")
    plt.xlabel("Std")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    #you can also interplot the map parameters --> distance_function(parameters) using your favorit regressor
    regressor = SVR(kernel='rbf', C=1, gamma= 'scale')
    interpolator = abc_sampling.Interpolator(history= history, regressor= regressor)
    interpolator.fit()


    plotter = abc_sampling.HistoryPlotter(history= history, model= interpolator)
    plotter.plot_interpolated_surface(grid_size=100)

if __name__ == "__main__":
    main()

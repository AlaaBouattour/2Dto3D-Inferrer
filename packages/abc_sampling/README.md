# ABC Sampling

**ABC Sampling** is a Python package for performing Approximate Bayesian Computation (ABC) using a rejection sampling approach. It allows users to infer parameters by comparing simulated data with observed data using a customizable distance function. The package also provides utilities for interpolating the parameter-to-distance mapping and visualizing the sampling history.

## Features

- **ABC Rejection Sampling:**  
  Implement a rejection sampling algorithm to explore the parameter space and infer parameters based on the discrepancy between simulated and observed data.

- **Flexible Prior Samplers:**  
  The package includes various prior samplers (e.g., Uniform, Beta) that generate candidate parameters.

- **Distance Functions:**  
  Built-in distance functions (such as the Wasserstein distance) are available to quantify the difference between simulated and observed datasets.

- **Interpolation and Visualization:**  
  Fit regression models (e.g., SVR) to the ABC sampling history to approximate the mapping from parameters to distance, and visualize this mapping with plotting utilities.

## Installation

This package is configured using a `pyproject.toml` file. To install the package in editable mode, follow these steps:

1. Navigate to the package directory (the folder containing `pyproject.toml`):
   ```bash
   cd biomasse/packages/abc_sampling
   ```
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

Below is a minimal example that demonstrates how to use the ABC Sampling package:

```python
import abc_sampling

# Assume you have a simulator that generates data and a prior sampler.
# For example, using a Gaussian simulator for demonstration:
from utils import GaussianSimulator

# Generate synthetic observed data using the Gaussian simulator.
true_params = [7.0, 1.5]  # e.g., mean = 7.0, std = 1.5
simulator = GaussianSimulator(true_params)
observed_data = simulator.generate_samples(1000)

# Define a uniform prior over the parameters (e.g., for mean and std).
prior = abc_sampling.priors.Uniform([0, 10, 0, 5])

# Define a distance function (e.g., Wasserstein distance)
distance_func = abc_sampling.distance_function

# Create an ABC Rejection Sampler.
abc_sampler = abc_sampling.RejectionSampling(
    simulator=simulator,
    dataAB=observed_data,
    prior_sampler=prior,
    distance_func=distance_func
)

# Run the sampler.
abc_sampler.run(N_iter=10000, epsilon=0.5)

# Retrieve accepted parameters and sampling history.
accepted_params = abc_sampler.get_accepted_params()
history = abc_sampler.get_history()
print("Accepted parameters:", accepted_params)
```

For an integrated example that uses the granulometre package to generate data, see the examples in the `examples/` folder (e.g., `example_1.py`).

## Project Structure

The package is organized as follows:

```
abc_sampling/
├── pyproject.toml           # Build configuration for the abc_sampling package.
├── README.md                # This file.
└── src/
    └── abc_sampling/
        ├── __init__.py      # Exposes main modules and classes.
        ├── abc.py           # Contains the RejectionSampling class and related ABC methods.
        ├── priors.py        # Contains prior samplers (e.g., Uniform, Beta).
        ├── utils.py         # Utility functions 
        
```

## Examples

The package includes several examples in the `examples/` folder:

- **example_0.py:** A toy example using a Gaussian simulator.
- **example_1.py:** An integrated example using the granulometre package to generate data and then performing ABC sampling.

## Contributing


## License

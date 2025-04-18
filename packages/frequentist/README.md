# frequentist

A simple Frequentist estimation package that allows you to define distance metrics (KDE-based, etc.) and 
use optimization methods (e.g., SciPy `minimize`, differential evolution, or custom PSO) to estimate parameters 
of a given generator (see the `granulometre` package).

## Overview

This package provides:

- **`frequentist.distance`**: Classes related to distance functions (e.g. `KDEDistance`).
- **`frequentist.optimizer`**: Classes related to optimization (e.g. `SciPyMinimizeOptimizer`, `SciPyDifferentialEvolutionOptimizer`, `PSOOptimizer`).
- **`frequentist.estimator`**: A simple wrapper (`FrequentistEstimator`) that combines a distance + optimizer to find the parameter vector that minimizes the chosen distance.

All classes are designed to be easily extended or replaced with custom implementations.


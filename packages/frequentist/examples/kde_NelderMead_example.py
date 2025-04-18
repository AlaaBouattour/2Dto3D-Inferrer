import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import granulometre
from granulometre.generator import Simulator

from frequentist import KDEDistance, FrequentistEstimator, Minimize

np.random.seed(42)

true_params = [1.0, 2.0]
dist_data = granulometre.distribution.Uniform(true_params)
sim_data_gen = Simulator(dist_data)
data_obs = sim_data_gen.generate_samples(100)
data_df = pd.DataFrame(data_obs, columns=["Dmax", "Dmin"])

dist_inference = granulometre.distribution.Uniform([0, 10])
sim_inference = Simulator(dist_inference)

distance = KDEDistance(simulator=sim_inference, bandwidth=0.2, n_sim=1000)
optimizer = Minimize(distance=distance, bounds=[(0, 10), (0, 10)], initial_guess=[1.0, 1.0])
estimator = FrequentistEstimator(distance=distance, optimizer=optimizer)

best_theta, best_val = estimator.fit(data_df.values)
print("Using KDEDistance")
print("Best theta:", best_theta)
print("Best distance:", best_val)

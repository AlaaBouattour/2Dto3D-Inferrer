import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import granulometre
from granulometre.generator import Simulator

from frequentist import KDEDistance, FrequentistEstimator, PSO

np.random.seed(42)

true_params = [1.0, 2.0]
dist_data = granulometre.distribution.Uniform(true_params)
sim_data_gen = Simulator(dist_data)
data_obs = sim_data_gen.generate_samples(100)
data_df = pd.DataFrame(data_obs, columns=["Dmax", "Dmin"])

dist_inference = granulometre.distribution.Uniform([0, 10])
sim_inference = Simulator(dist_inference)

distance = KDEDistance(simulator=sim_inference, bandwidth=0.2, n_sim=1000)

optimizer = PSO(
    distance=distance,
    bounds=[(1e-3, 10), (1e-3, 10)], 
    n_particles=30,
    max_iter=100,
    w=0.5,
    c1=1.2,
    c2=2.0,
    seed=42
)

estimator = FrequentistEstimator(distance=distance, optimizer=optimizer)

best_theta, best_val = estimator.fit(data_df.values)
print("Using KDEDistance with PSO")
print("Best theta:", best_theta)
print("Best distance:", best_val)

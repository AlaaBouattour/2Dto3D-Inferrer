import numpy as np

class FrequentistEstimator:
    """
    Simple wrapper around distance + optimizer to produce a frequentist estimate of theta.
    """
    def __init__(self, distance, optimizer):
        """
        distance : Distance instance (defines evaluate())
        optimizer : an optimizer instance (must define optimize(data)->(best_theta, best_val))
        """
        self.distance = distance
        self.optimizer = optimizer
        self.best_theta_ = None
        self.best_distance_ = None

    def fit(self, data):
        best_pos, best_val = self.optimizer.optimize(data)
        self.best_theta_ = best_pos
        self.best_distance_ = best_val
        return self.best_theta_, self.best_distance_
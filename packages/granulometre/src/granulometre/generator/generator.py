import numpy as np
from ..projection.projection import ellipsoid_projection_axes
from . import distributions as distributions


class Simulator:
    def __init__(self, distribution: distributions.Distribution):
        """
        Simulates 3D ellipsoids (axes a >= b >= c) using a 'Distribution' object.
        Then applies a random orientation (via 'ellipsoid_projection_axes'),
        and retrieves the projected ellipse axes (A,B).

        - distribution: an instance of Distribution (e.g., UniformDistribution, BetaDistribution, etc.)
        """
        self.distribution = distribution
        self.angles = None

    def set_params(self, new_params):
        """
        Dynamically update the distribution parameters (useful for ABC rejection).
        """
        self.distribution.set_params(new_params)

    def _draw_3d_axes(self):
        """
        Samples axes (a,b,c) from the distribution object,
        assumed to be already in descending order.
        """
        return self.distribution.sample()

    def generate_samples(self, N: int):
        """
        Generates N ellipsoids by:
          - sampling (a,b,c) in descending order,
          - projecting them in 2D to obtain (A,B).
        Returns two arrays (A_list, B_list) of length N.
        """
        A_list = []
        B_list = []
        angles = []
        for _ in range(N):
            a, b, c = self._draw_3d_axes()
            A_proj, B_proj, angle = ellipsoid_projection_axes(a, b, c)
            A_list.append(A_proj)
            B_list.append(B_proj)
            angles.append(angle)
        self.angles = angles
        return np.column_stack((A_list, B_list))


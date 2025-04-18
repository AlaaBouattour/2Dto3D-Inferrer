import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from scipy.optimize import minimize, differential_evolution

class Base(ABC):
    """
    Base class for all optimizers used to minimize a Distance function.
    """

    @abstractmethod
    def optimize(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Minimizes Distance.evaluate(theta, data).
        Returns the best-found theta and the corresponding distance value.
        """
        pass

class Minimize(Base):
    """
    Uses scipy.optimize.minimize(...) to perform local minimization.
    By default, uses Nelder-Mead, which does not require derivatives,
    and ignores bounds. To enforce bounds, switch method (e.g., 'L-BFGS-B').
    """
    def __init__(
        self,
        distance,
        bounds: Optional[List[Tuple[float, float]]] = None,
        initial_guess: Optional[np.ndarray] = None,
        method: str = "Nelder-Mead",
        options: dict = None
    ):
        """
        Parameters
        ----------
        distance : a Distance instance
            Must have .evaluate(theta, data) -> float
        bounds : list of (low, high) pairs or None
            If using a bounded method like L-BFGS-B, these define parameter constraints
        initial_guess : initial parameter guess (if None and bounds provided, uses midpoints)
        method : str
            E.g. "Nelder-Mead", "Powell", "L-BFGS-B", etc.
        options : dict
            Extra arguments for scipy.optimize.minimize
        """
        self.distance = distance
        self.bounds = bounds
        self.initial_guess = initial_guess
        self.method = method
        self.options = options if options is not None else {}

    def optimize(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Minimizes distance.evaluate(theta, data).
        Returns (best_theta, best_distance_value).
        """
        if self.initial_guess is None:
            if self.bounds is not None and len(self.bounds) > 0:
                mid = []
                for low, high in self.bounds:
                    mid.append(0.5 * (low + high))
                initial_guess = np.array(mid)
            else:
                raise ValueError("No initial guess or bounds provided.")
        else:
            initial_guess = np.array(self.initial_guess)

        def objective(theta):
            return self.distance.evaluate(theta, data)

        effective_bounds = self.bounds
        if self.method in ["Nelder-Mead", "Powell"]:
            effective_bounds = None

        result = minimize(
            objective,
            x0=initial_guess,
            method=self.method,
            bounds=effective_bounds,
            options=self.options
        )
        return result.x, result.fun

class DifferentialEvolution(Base):
    """
    Global, derivative-free minimization using scipy.optimize.differential_evolution.
    Suitable for complex landscapes and multi-modal functions.
    """
    def __init__(
        self,
        distance,
        bounds: List[Tuple[float, float]],
        maxiter: int = 100,
        popsize: int = 15,
        seed: Optional[int] = None,
        strategy: str = "best1bin"
    ):
        """
        distance : Distance instance
        bounds : list of (low, high) pairs for each parameter dimension
        maxiter : max number of DE iterations
        popsize : population size factor
        seed : optional random seed for reproducibility
        strategy : DE strategy (e.g. 'best1bin', 'rand1bin', etc.)
        """
        self.distance = distance
        self.bounds = bounds
        self.maxiter = maxiter
        self.popsize = popsize
        self.seed = seed
        self.strategy = strategy

    def optimize(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Runs differential evolution to minimize distance.evaluate.
        Returns (best_theta, best_distance_value).
        """
        def objective(theta):
            return self.distance.evaluate(theta, data)
        result = differential_evolution(
            func=objective,
            bounds=self.bounds,
            strategy=self.strategy,
            maxiter=self.maxiter,
            popsize=self.popsize,
            seed=self.seed
        )
        return result.x, result.fun

class PSO(Base):
    """
    A simple custom Particle Swarm Optimization (PSO).
    For demonstration, not heavily optimized.
    """
    def __init__(
        self,
        distance,
        bounds: List[Tuple[float, float]],
        n_particles: int = 20,
        max_iter: int = 100,
        w: float = 0.5,
        c1: float = 1.0,
        c2: float = 2.0,
        seed: Optional[int] = None
    ):
        """
        distance : Distance instance
        bounds : list of (low, high) pairs for each dimension
        n_particles : size of the swarm
        max_iter : max iteration for updating swarm
        w : inertia coefficient
        c1 : cognitive component
        c2 : social component
        seed : optional random seed for reproducibility
        """
        self.distance = distance
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(seed=seed)

    def optimize(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Minimizes distance.evaluate(theta, data) using a simple PSO approach.
        Returns (best_theta, best_distance_value).
        """
        dim = len(self.bounds)
        swarm_pos = np.zeros((self.n_particles, dim))
        swarm_vel = np.zeros((self.n_particles, dim))
        for i in range(dim):
            low, high = self.bounds[i]
            swarm_pos[:, i] = self.rng.uniform(low, high, size=self.n_particles)
            swarm_vel[:, i] = self.rng.uniform(-abs(high - low), abs(high - low), size=self.n_particles)

        pbest_pos = swarm_pos.copy()
        pbest_val = np.array([self.distance.evaluate(pos, data) for pos in pbest_pos])

        gbest_idx = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_val = pbest_val[gbest_idx]

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                swarm_vel[i] = (
                    self.w * swarm_vel[i]
                    + self.c1 * r1 * (pbest_pos[i] - swarm_pos[i])
                    + self.c2 * r2 * (gbest_pos - swarm_pos[i])
                )
                swarm_pos[i] += swarm_vel[i]
                for d in range(dim):
                    low, high = self.bounds[d]
                    if swarm_pos[i, d] < low:
                        swarm_pos[i, d] = low
                    elif swarm_pos[i, d] > high:
                        swarm_pos[i, d] = high

                val = self.distance.evaluate(swarm_pos[i], data)
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest_pos[i] = swarm_pos[i].copy()
                    if val < gbest_val:
                        gbest_val = val
                        gbest_pos = swarm_pos[i].copy()

        return gbest_pos, gbest_val

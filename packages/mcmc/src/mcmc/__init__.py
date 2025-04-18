from . import prior
from . import likelihood

from .sampler import Sampler

# alias
Prior = prior.Prior
UniformPrior = prior.Uniform

Likelihood = likelihood.Likelihood
KDELikelihood = likelihood.KDE
KLLikelihood = likelihood.KL

# from mcmc import *
__all__ = [
    "prior",           # enable:  mcmc.prior.X
    "likelihood",      # enable:  mcmc.likelihood.X
    "Prior",           # enable:  mcmc.Prior(...)
    "UniformPrior",    # enable:  mcmc.UniformPrior(...)
    "Likelihood",      # enable:  mcmc.Likelihood(...)
    "KDELikelihood",
    "KLLikelihood",
    "Sampler",
]

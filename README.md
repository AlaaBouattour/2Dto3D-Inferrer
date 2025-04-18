# Biomasse

**Biomasse** is a research project aimed at estimating the parameters of the distribution of 3D particle characteristics from their 2D projections obtained via a granulometer. The main objective is to develop a robust inference method to derive three-dimensional properties from two-dimensional measurements.

The project currently uses a model based on 3D ellipsoids as a first approach, but aims to develop more complex and realistic modeling in the future.

The project is organized into several independent packages:

- **abc_sampling**: Contains methods for Approximate Bayesian Computation (ABC), rejection sampling, interpolation, and other statistical utilities.
- **granulometre**: Provides modules for simulating the granulometer, including generation, rotation, and projection of 3D particles.
- **mcmc**: Contains Markov Chain Monte Carlo (MCMC) sampling methods used for parametric inference.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

### 1. Clone the Repository

```bash
git clone <repository_URL>
cd biomasse
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

- **On Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **On Linux/Mac:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install the Packages in Editable Mode

This project is organized into separate packages. You must install each package in editable mode.

```bash
cd biomasse
pip install -e .\granulometre
pip install -e .\abc_sampling
pip install -e .\mcmc
```

## Usage

After installation, you can import the packages in your scripts or notebooks. For example, to test the mcmc method in `tests/test_mcmc.ipynb` notebook, follow these steps:


   - Install `ipykernel` in your virtual environment:
     ```bash
     pip install ipykernel
     ```
   - Then, register your virtual environment as a Jupyter kernel:
     ```bash
     python -m ipykernel install --user --name biomasse-env --display-name "Python (biomasse-env)"
     ```
   - In your notebook, select the kernel **"Python (biomasse-env)"** via the Kernel menu.


Then, you can import the packages as follows:
```python
from granulometre import GranulometreSimulator, distributions
from mcmc import (
    MCMCSampler,
    UniformPrior,
    KLLikelihood,
    KDELikelihood

) 
```

You can also run test scripts such as `tests/test_libs_abc.py` to see how to use our packages to estimate the parameters of the 3D particle distribution using the ABC rejection sampling method.

**Typical Workflow:**
1. Generate synthetic 3D particles according to a parametric distribution.
2. Project these particles to obtain 2D characteristics.
3. Compare with observed 2D data.
4. Infer the parameters of the 3D distribution.

## Contributing


## License


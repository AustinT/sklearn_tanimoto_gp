# Sklearn Tanimoto GP Kernels

This package provides Tanimoto similarity kernels compatible with scikit-learn's Gaussian Process module (`sklearn.gaussian_process.kernels.Kernel`).

Intended usage:

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn_tanimoto_gp import Tanimoto

# Example Data
X_train = np.random.rand(10, 5) * 10 # Example non-negative data
y_train = np.sum(X_train[:, :2], axis=1) + np.random.randn(10) * 0.1
X_test = np.random.rand(5, 5) * 10

# Using MinMaxTanimoto (or Tanimoto)
gp = GaussianProcessRegressor(kernel=Tanimoto(), alpha=1e-5, normalize_y=True)
gp.fit(X_train, y_train)
y_pred, sigma = gp.predict(X_test, return_std=True)
```

## Installation

```bash
pip install git+https://github.com/AustinT/sklearn_tanimoto_gp.git
```

## Development

PRs are welcome. Please use pre-commit and run tests.

NOTE: LLMs were used in the initial development of this package.
I have checked the outputs but cannot guarantee that there are no
mistakes.

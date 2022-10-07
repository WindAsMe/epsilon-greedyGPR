from sklearn.datasets import load_boston
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK


boston = load_boston()
boston_X = boston.data
boston_y = boston.target

train_set = np.random.choice([True, False], len(boston_y), p=[.75, .25])


mixed_kernel = CK(1.0, (1e-4, 1e4)) * RBF(10, (1e-4, 1e4))
gpr = GaussianProcessRegressor(alpha=5, n_restarts_optimizer=20, kernel=mixed_kernel)

gpr.fit(boston_X[train_set], boston_y[train_set])
test_preds = gpr.predict(boston_X[train_set])

print(test_preds)
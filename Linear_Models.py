# Linear Models are a class of models that are widely used in practice
# and have been studied extensively in the last few decades, with roots
# going back over a hundred years.

import sys
import os
import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import mglearn
from sklearn.linear_model import LinearRegression

mglearn.plots.plot_linear_regression_wave()
X, y = mglearn.datasets.make_wave(n_samples = 60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
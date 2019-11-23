# Jorge Bonilla

# Packages

import sys
import os
import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import mglearn
# Chapter 2 - Supervised Learning

# Supervised learning is used whenever we want to predict a certain outcome from a given
# input and we have examples of input/output pairs.

# The goal is to make accurate predictions for new, never-before-seen data.

# Supervised Machine Learning Algorithms

# Generate Dataset

X, y = mglearn.datasets.make_forge()

# Plot Dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc = 4)
plt.xlabel("First Feature")
plt.ylabel("Second Feature")
print("X.shape: {}".format(X.shape))
plt.show()
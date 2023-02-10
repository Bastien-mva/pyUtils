import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd

import os

from pyDeepInsight.transformer import ImageTransformer
from pyDeepInsight.scaler import insightMinMaxScaler

Y = pd.read_csv("data/Y_test.csv")
sc = insightMinMaxScaler()
Y = sc.fit_transform(Y)
print(Y)
transform = ImageTransformer(pixels=22, random_state=1, n_channels=6)
X = transform.fit_transform(Y, plot=False)
transform.show_fdm(n_channel=1)
plt.show()
transform.pixels = 20
transform.show_fdm(n_channel=1)
plt.show()

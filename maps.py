
import numpy as np
import pandas as pd
from math import sqrt

import pickle
import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib_inline
from IPython.display import set_matplotlib_formats
from sklearn.metrics import r2_score
matplotlib_inline.backend_inline.set_matplotlib_formats('retina')


## for machine learning

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Masking
from keras.layers import LeakyReLU
from keras.optimizers import adam_legacy,adamax_legacy
from keras_tuner.tuners import BayesianOptimization


#file_path = "datasets/SA-1923-2023/" + "maps" + ".csv"
file_path = "datasets/NorthAnatolian-1923-2023/" + "maps" + ".csv"

df_eq = pd.read_csv(file_path)

df_eq = df_eq[["time","mag_max", "event_count", "mag_mean", "mag_sum", "longitude_mean", "latitude_mean", "depth_mean", "time_diff_mean", "time_diff_std", "depth_std", "latitude_std", "longitude_std", "mag_scatter", "weight_richter", "weight_Mw"]]
df_eq["time"] = pd.to_numeric(pd.to_datetime(df_eq["time"]))


#sns.kdeplot(df_eq["mag_max"], df_eq["depth_mean"])
#df_eq.plot(subplots=True,figsize=(14,14));

corr = df_eq.corr()

#sns.heatmap(corr)
sns.distplot(df_eq["mag_max"],axlabel="Earthquakes on  NAFZ")
plt.ylabel("Scatter")
#sns.countplot(data=df_eq)

plt.legend()
#plt.legend()
plt.show()


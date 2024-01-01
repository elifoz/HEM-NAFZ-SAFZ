

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


file_path = "datasets/SA-1923-2023/" + "combined_SA" + "_timeseries" + ".csv"

df_eq = pd.read_csv(file_path)

df_eq = df_eq[["time", "mag", "sig", "longitude", "latitude", "depth"]]

df_eq["time"] = pd.to_datetime(df_eq["time"])
df_eq["timestamps"] = df_eq["time"]
# df_eq.set_index('time', inplace=True)

print(df_eq)
df_eq.info()

df_eq[df_eq.duplicated('timestamps')] #need to drop repetitive rows (10 rows). There were some overlaps in original data.
print(df_eq)

# Dropping rows
df_eq = df_eq.drop_duplicates(subset=['timestamps'])

df_eq["timestamps"].diff().describe()

df_eq.info()

#FEATURE ENGINEERING
# Number 1: Time intervals between consecutive earthquakes.
df_eq["time_diff"] = df_eq["timestamps"].diff()

df_eq["time_diff_float"] = df_eq["time_diff"].apply(lambda x: x.total_seconds())

#Number 2: the rolling of magnitudes from the last 10 earthquakes

df_eq["mag_roll_10"] = df_eq["mag"].rolling(window=10).mean()

df_eq.dropna(inplace = True)
print(df_eq.describe().T)

df_eq[df_eq["time_diff_float"] > 86400].shape
print(df_eq[df_eq["time_diff_float"] > 86400]) # arasında 1 günden fazla olan deprem sayısı [5342 rows x 10 columns]

min_date = df_eq["time"].min().date()

print(min_date) #en eski tarih 1926-06-29

max_date = df_eq["time"].max().date()
print(max_date) # en yeni tarih 2023-08-28

start_date = min_date

number_of_days = (max_date - min_date).days
print("number_of_days", number_of_days) # deprem olan toplam 35489 gün
date_list = []
for day in range(number_of_days):
  a_date = (start_date + datetime.timedelta(days = day))
  date_list.append(a_date)

print(date_list[1], type(date_list[0]))

print(df_eq)

df_eq.set_index('time', inplace=True)

df_daily = pd.DataFrame()

# AS is year-start frequency

df_daily['mag_max'] = df_eq.mag.resample('D').max() # What is the max of events
df_daily['event_count'] = df_eq.mag.resample('D').count() #How many events happened
df_daily['mag_mean'] = df_eq.mag.resample('D').mean() #What is the mean of values for that particular day
df_daily['mag_sum'] = df_eq.mag.resample('D').sum() # What is the sum of moments
df_daily['mag_scatter'] = (df_eq.mag.resample('D').std()) # What is the scatter (dağılım) of event magnitudes
df_daily["mag_roll_10"] = df_daily["mag_mean"].rolling(window=10).mean() #son 10 depremin büyüklüklerinin yuvarlanması

df_daily['longitude_mean'] = df_eq.longitude.resample('D').mean() #Mean location of events
df_daily['longitude_std'] = df_eq.longitude.resample('D').std() #Std location of events

df_daily['latitude_mean'] = df_eq.latitude.resample('D').mean() #Mean location of events
df_daily['latitude_std'] = df_eq.latitude.resample('D').std() #Std location of events

df_daily['depth_mean'] = df_eq.depth.resample('D').mean() #Mean location of events
df_daily['depth_std'] = df_eq.depth.resample('D').std() #Std location of events

df_daily['time_diff_float_mean'] = df_eq.time_diff_float.resample('D').mean() #Event spacing
df_daily['time_diff_float_std'] = df_eq.time_diff_float.resample('D').std() #Std location of events

print(df_daily)
print(df_daily.info())

plt.figure(figsize = (8, 5))

#sns.heatmap(df_daily.isnull(), cbar=False) #Isı haritaları, birden çok değişken arasında varyans göstermek, herhangi bir tasarım ortaya çıkarmak, herhangi bir değişkenin birbirine benzer olup olmadığını göstermek ve aralarında herhangi bir korelasyon olup olmadığını tespit etmek için kullanılabilir.


df_daily_clean = df_daily[df_daily.index > "1973-01-01"]

df_daily_clean.drop(columns = ["mag_roll_10"], inplace = True)

print(df_daily_clean)
plt.figure(figsize = (8, 5))

#sns.heatmap(df_daily_clean.isnull(), cbar=False) #Isı haritaları, birden çok değişken arasında varyans göstermek, herhangi bir tasarım ortaya çıkarmak, herhangi bir değişkenin birbirine benzer olup olmadığını göstermek ve aralarında herhangi bir korelasyon olup olmadığını tespit etmek için kullanılabilir.


print(df_daily_clean.info()) #temizlendikten sonraki veriler 18501

#Source: https://medium.com/@drnesr/filling-gaps-of-a-time-series-using-python-d4bfddd8c460
df_daily_clean.interpolate('time', inplace = True) #kayıp değerlerin hesaplanması
print(df_daily_clean)
file_path = "datasets/SA-1923-2023/" + "df_daily_clean" + ".csv"
#df_daily_clean.to_csv(file_path)

df_daily_clean_revized= df_daily_clean.assign(missing= np.nan)
df_daily_clean_revized.missing[df_daily_clean_revized.time_diff_float_std.isna()] = df_daily_clean_revized.time_diff_float_mean
df_daily_clean_revized.info()
#isna() func: Dizi benzeri bir nesne için eksik değerleri algılar.

#df_daily_clean_revized.plot(style=['k--', 'bo-', 'r*'], figsize=(20, 10));
"""
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolateLinear=df_daily_clean_revized.time_diff_float_std
.interpolate(method='linear'))
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolateTime=df_daily_clean_revized.time_diff_float_std
.interpolate(method='time'))
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolateQuadratic=df_daily_clean_revized.time_diff_float_std
.interpolate(method='quadratic'))
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolateCubic=df_daily_clean_revized.time_diff_float_std
.interpolate(method='cubic'))
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolateSLinear=df_daily_clean_revized.time_diff_float_std
.interpolate(method='slinear'))
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolateAkima=df_daily_clean_revized.time_diff_float_std
.interpolate(method='akima'))
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolatePoly5=df_daily_clean_revized.time_diff_float_std
.interpolate(method='polynomial', order=5))
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolatePoly7=df_daily_clean_revized.time_diff_float_std
.interpolate(method='polynomial', order=7))
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolateSpline3=df_daily_clean_revized.time_diff_float_std
.interpolate(method='spline', order=3))
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolateSpline4=df_daily_clean_revized.time_diff_float_std
.interpolate(method='spline', order=4))
df_daily_clean_revized = df_daily_clean_revized.assign(InterpolateSpline5=df_daily_clean_revized.time_diff_float_std
.interpolate(method='spline', order=5))
#bu kısımda hangi interpolasyon yöntemi daha iyi ona bakıyoruz
results = [(method, r2_score(df_daily_clean_revized.time_diff_float_mean, df_daily_clean_revized[method])) for method in list(df_daily_clean_revized)[3:]]
results_df = pd.DataFrame(np.array(results), columns=['Method', 'R_squared'])
#print(results_df.sort_values(by='R_squared', ascending=False))
print(results_df)
"""
# Filling using mean or median
# Creating a column in the dataframe
# instead of : df['NewCol']=0, we use
# df = df.assign(NewCol=default_value)
# to avoid pandas warning.
df_daily_clean_revized = df_daily_clean_revized.assign( TimeDifFloatStdNew =df_daily_clean_revized.time_diff_float_std.fillna(df_daily_clean_revized.time_diff_float_std.mean()))
df_daily_clean_revized = df_daily_clean_revized.assign( DepthStdNew =df_daily_clean_revized.depth_std.fillna(df_daily_clean_revized.depth_std.mean()))
df_daily_clean_revized = df_daily_clean_revized.assign( LatitudeStdNew =df_daily_clean_revized.latitude_std.fillna(df_daily_clean_revized.latitude_std.mean()))
df_daily_clean_revized = df_daily_clean_revized.assign( LongitudeStdNew =df_daily_clean_revized.longitude_std.fillna(df_daily_clean_revized.longitude_std.mean()))
df_daily_clean_revized = df_daily_clean_revized.assign( MagScatterNew =df_daily_clean_revized.mag_scatter.fillna(df_daily_clean_revized.mag_scatter.mean())) #ortalamaya göre boşluk doldurma
#df_daily_clean_revized = df_daily_clean_revized.assign(FillMedian=df_daily_clean_revized.target.fillna(df_daily_clean_revized.target.median()))
df_daily_clean_revized.drop(columns = ["mag_scatter", "longitude_std", "latitude_std", "depth_std", "time_diff_float_std", "missing" ], inplace = True) #eksik sütunlar

#df_daily_clean_revized.to_csv(file_path)

print(df_daily_clean_revized.info())

df_eq = df_daily_clean_revized
print(df_eq.shape)

## Large earthquakes labeling:
label = []
cnt = 0
for i, mag in enumerate(df_eq["mag_max"]):
    if (mag>5.5):
        cnt = cnt + 1
        label.append(int(cnt))
    else:
        label.append(0)

df_eq["large_eq_label"] = label

print(df_eq.describe().T)

large_eq = df_eq[df_eq["large_eq_label"]>0]

#ZAMAN FARKI - DEPREM SCATTER GRAFİĞİ
"""
ax = plt.gca()
large_eq.plot(kind="scatter", x="time_diff_float_mean", y="mag_max",
     label="D",
    c=large_eq.index, cmap=plt.get_cmap("jet"),
    colorbar=True, alpha=0.4, figsize=(10,7), ax = ax
)
plt.legend()
plt.tight_layout()
"""
#KORELASYON İLİŞKİSİ
"""
df_corr = df_eq.corr()
#plt.figure(figsize=(6,6))
plt.figure(figsize=(8,6))
sns.heatmap(df_corr,
        vmin=-1,
        cmap='coolwarm',
        annot=True);
"""
#DERİNLİK - ENLEM & BOYLAM İLİŞKİSİ
"""
df_eq_plot = df_eq[df_eq["depth_mean"]<50]
plt.hist2d(df_eq_plot['longitude_mean'], df_eq_plot['latitude_mean'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')

"""
#DERİNLİK-DEPREM BÜYÜKLÜĞÜ İLİŞKİSİ
"""
df_eq_plot = df_eq[df_eq["depth_mean"]<50]

plt.hist2d(df_eq_plot['mag_max'], df_eq_plot['depth_mean'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('EQ magnitude')
plt.ylabel('Depth (km)')
"""
df_eq_plot = df_eq.copy()
df_eq_plot["time_days"] = (np.array(df_eq_plot["time_diff_float_mean"])/86_400)
plt.figure(figsize = (14, 10));
df_eq_plot["time_days"].plot.hist(grid=True, bins=100, rwidth=0.9,orientation="horizontal",
                   color='purple')
plt.title('San Andreas - son 50 yılda gerçekleşen depremler',fontsize = 24)
plt.ylabel('Zaman aralığı (gün)',fontsize = 22)
plt.xlabel('Depremin meydana gelme sıklığı',fontsize = 22)

plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)

plt.grid(axis='y', alpha=0.75)
plt.xscale('log')


plt.show()
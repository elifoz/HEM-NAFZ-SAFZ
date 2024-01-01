import requests
import json
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
## for geospatial
import folium
import geopy
from mpl_toolkits.basemap import Basemap
from IPython.display import set_matplotlib_formats
## for machine learning
from sklearn import preprocessing, cluster

set_matplotlib_formats('retina')

file_path = "datasets/NorthAnatolian-1923-2023/" + "combined_NAFZ" + "_timeseries" + ".csv"

#Converting to time-series with respect to "time" column
df_eq = pd.read_csv(file_path, index_col=0)

df_eq["time"] = df_eq.index

df_eq.head(3)


df_eq.info()

#%config InlineBackend.figure_format = 'retina'

plt.figure(figsize = [7, 4])
plt.hist(np.array(df_eq.mag), log=True, color="#FF66FF");
plt.xlabel("Deprem Büyüklüğü", fontsize = 12);
plt.ylabel("Frekans", fontsize = 12);

plt.figure(figsize = [7, 4])
plt.hist(np.array(df_eq.depth), log=True, color="#3399FF");
plt.xlabel("Deprem Derinliği", fontsize = 12);
plt.ylabel("Frekans", fontsize = 12);


plt.figure(figsize = [7, 4])
plt.hist(np.array(df_eq.longitude), log=True, color="#FF8000");
plt.xlabel("Boylam", fontsize = 12);
plt.ylabel("Frekans", fontsize = 12);


plt.figure(figsize = [7, 4])
plt.hist(np.array(df_eq.latitude), log=True, color="#009900");
plt.xlabel("Enlem", fontsize = 12);
plt.ylabel("Frekans", fontsize = 12);


df_eq_large = df_eq[df_eq["mag"]>6].copy()

# Fixing the time column datatype
df_eq_large["time"] = pd.to_datetime(df_eq_large["time"])

df_eq_large["time_diff_day"] = df_eq_large["time"].diff()

df_eq_large.head(5)


df_eq_large.info()

print(len(df_eq_large))

print(df_eq_large["time_diff_day"].describe())

plt.figure(figsize = [7, 4])
df_eq_large["time_diff_day"].astype('timedelta64[s]').plot.hist(color="#FF3333") #Burayı değiştirebilirsin belki bak

plt.xlabel("Zaman Farkı(Gün)", fontsize = 13);
plt.ylabel("Frekans", fontsize = 13);

# Pandas series with magnitudes greater than 6
plt.figure(figsize=(25,5))
plt.title("")
plt.ylabel("Büyüklük")
plt.xlabel("Tarih")
plt.plot(df_eq_large["time"],df_eq_large["mag"],marker="o", color="red")
# Tarih etiketlerini daha iyi bir yerleşime sahip hale getir
plt.gcf().autofmt_xdate() #değerleri çapraz şekilde koyma
plt.show()


plt.figure(figsize = [7, 4])
plt.scatter(df_eq_large["time_diff_day"].astype('timedelta64[s]'),
            df_eq_large.mag)
plt.plot(df_eq_large["time_diff_day"],df_eq_large["time_diff_day"])

plt.xlabel("EQ magnitude", fontsize = 13);
plt.ylabel("Frequency", fontsize = 13);

ax = plt.gca()

df_eq_large.plot(kind="scatter", x="longitude", y="latitude",
    s=df_eq_large['mag']/0.05, label="Large EQ",
    alpha=0.4, figsize=(10,7), ax = ax
)
plt.legend()

# plt.savefig("../plots/EDA_long_lat.png")


plt.tight_layout()

type(list(pd.to_datetime(df_eq_large.index))[0].date())

fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='h', 
            lat_0=38, lon_0=-120,
            width=2.0E6, height=2.0E6)

# m.bluemarble()
m.etopo()

# m.drawcoastlines(color='blue',linewidth=3)
# m.drawcountries(color='gray',linewidth=3)
m.drawstates(color='gray')


lon = list(df_eq_large["longitude"])
lat = list(df_eq_large["latitude"])

m.scatter(lon,lat, latlon=True,
#           c=color,
          s=df_eq_large['mag']/0.05,
          c = 'red',
#           cmap='YlGnBu_r',
          alpha=0.5)

# df_eq_large.plot(kind="scatter", x="longitude", y="latitude",
#     s=df_eq_large['mag']/0.05, label="Large EQ",
#     alpha=0.4, figsize=(10,7), ax = ax
# )

# Source: https://towardsdatascience.com/clustering-geospatial-data-f0584f0b04ec

region = "NAFZ"
## get location
locator = geopy.geocoders.Nominatim(user_agent="MyCoder")
location = locator.geocode(region)
print(location)
## keep latitude and longitude only
location = [location.latitude, location.longitude]
print("[lat, long]:", location)

x, y = "latitude", "longitude"
color = "time"
size = "mag"
popup = "name"
data = df_eq_large.copy()

## create color column
lst_colors=["black"]
data["color"] = "black"
lst_elements = sorted(list(data["color"].unique()))

## create size column (scaled)
scaler = preprocessing.MinMaxScaler(feature_range=(3,15))
data["size"] = scaler.fit_transform(
               data[size].values.reshape(-1,1)).reshape(-1)




## initialize the map with the starting location
map_ = folium.Map(location=location, tiles="cartodbpositron",
                  zoom_start=7)
## add points
data.apply(lambda row: folium.CircleMarker(
           location=[row[x],row[y]], popup=row[popup],
           color=row["color"], fill=True,
           radius=row["size"]).add_to(map_), axis=1)

#add html legend
legend_html = """<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">&nbsp;<b>""" +color+""" :</b><br>"""

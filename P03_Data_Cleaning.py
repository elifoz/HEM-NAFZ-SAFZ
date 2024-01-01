
import pandas as pd
#Importing the built-in function, prior to that added the assets path to the system path

import sys
# inserting the parent directory into current path
sys.path.insert(1, 'main')

from EQ_USGS import clean_EQ_USGS_df

import glob
import os

#path = 'datasets/NorthAnatolian-1923-2023/' #bu klasörün altındakileri temizle
path = 'datasets/NorthAnatolian-1923-2023/' #bu klasörün altındakileri temizle

my_files = []

for file in sorted(os.listdir(path)):
    if file.endswith("newraw.csv"):
        my_files.append(path+file)
        
my_files[:10]

#Source: https://www.freecodecamp.org/news/how-to-combine-multiple-csv-files-with-8-lines-of-code-265183e0854/
df_combined = pd.concat([pd.read_csv(f) for f in my_files ])

df_combined.reset_index(drop=True, inplace=True)

df_combined.shape

df_combined.head(5)

df_combined_clean = clean_EQ_USGS_df(df_combined)

#file_path = "datasets/NorthAnatolian-1923-2023/" + "combined_NAFZ" + "_clean" + ".csv"
file_path = "datasets/NorthAnatolian-1923-2023/" + "combined_NAFZ" + "_clean" + ".csv"
df_combined_clean.to_csv(file_path)

df_combined_clean.head()

#file_path = "datasets/NorthAnatolian-1923-2023/" + "combined_NAFZ" + "_clean" + ".csv"
file_path = "datasets/NorthAnatolian-1923-2023/" + "combined_NAFZ" + "_clean" + ".csv"

df_eq = pd.read_csv(file_path)

# Dropping unnecessary columns
df_eq.drop(columns = "Unnamed: 0", inplace = True)

# Fixing the time column datatype
df_eq["time"] = pd.to_datetime(df_eq["time"])

#Adding name column for Folium map pop-ups
df_eq["name_mag"] = df_eq["mag"].apply(lambda x: "M: " + str(x) + " / ")
df_eq["name_date"] = df_eq["time"].apply(lambda x: " " + str(x.date())+ " / ")
df_eq["name"] = df_eq["name_mag"] + df_eq["name_date"] + df_eq["place"]
df_eq.drop(columns=["name_mag", "name_date"], inplace = True)

# Sorting the dataframe with respect to time
df_eq = df_eq.sort_values(by="time")

#Converting to time-series with respect to "time" column
df_eq.set_index('time', inplace=True)

# There were only some depth values missing, so imputing them, not a big deal of imputing processes
df_eq.fillna(df_eq.mean(), inplace = True)

df_eq.head(5)

len(df_eq)

#Source: https://stackoverflow.com/questions/13035764/remove-rows-with-duplicate-indices-pandas-dataframe-and-timeseries

df_eq = df_eq[~df_eq.index.duplicated(keep='first')]

len(df_eq)

# Finally save the dataframe as a time-series csv file
#file_path = "datasets/NorthAnatolian-1923-2023/" + "combined_NAFZ" + "_timeseries" + ".csv"
file_path = "datasets/NorthAnatolian-1923-2023/" + "combined_NAFZ" + "_timeseries" + ".csv"
df_eq.to_csv(file_path)

print("finished!")
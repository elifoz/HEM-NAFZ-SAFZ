import requests
import json
import numpy as np
import pandas.compat as pd
import pandas as pd
import matplotlib.pyplot as plt
import sys

# inserting the parent directory into current path
sys.path.insert(1, 'datasets/')

from EQ_USGS import get_EQ_USGS_df

Params = {}
Params["get_format"] = "geojson"  # Format for importing data

Params["min_date"] = "1923-01-01"  # Minimum date for reporting the data
Params["max_date"] = "2023-12-31"  # Maximum date for reporting the data

Params["min_magnitude"] = "2"  # Minimum magnitude of the reporting data
Params["max_magnitude"] = "10"  # Maximum magnitude of the reporting data

Params["min_latitude"] = "33"  # Minimum latitude-North Anatolian
Params["max_latitude"] = "40"  # Maximum latitude-North Anatolian

Params["min_longitude"] = "-124"  # Minimum longitude-North Anatolian
Params["max_longitude"] = "-112"  # Maximum longitude-North Anatolian

Params["order_by"] = "time"  # Ordering the data by parameters
Params["limit_data"] = "20000"  # Maximum number of data

df = get_EQ_USGS_df(Params)

file_path = "datasets/SA-1923-2023/" + "SA" + "_raw" + ".csv"

df.to_csv(file_path)

# %config InlineBackend.figure_format = 'retina'

plt.figure(figsize=[8, 5])

# plt.plot(df.time,df.mag);

# Format title and axis labels
plt.title("EQ parameters", fontsize=15)

plt.xlabel("Time", fontsize=12)
plt.ylabel("Magnitude", fontsize=12)

for start_num in range(1923, 2024, 1):
    # First quarter of the year

    start_str = f"""{start_num}-01-02"""
    end_str = f"""{start_num}-03-01"""

    Params = {}
    Params["get_format"] = "geojson"  # Format for importing data

    Params["min_date"] = start_str  # Minimum date for reporting the data
    Params["max_date"] = end_str  # Maximum date for reporting the data

    Params["min_magnitude"] = "2"  # Minimum magnitude of the reporting data
    Params["max_magnitude"] = "10"  # Maximum magnitude of the reporting data

    Params["min_latitude"] = "33"  # Minimum latitude-North Anatolian
    Params["max_latitude"] = "40"  # Maximum latitude-orth Anatolian

    Params["min_longitude"] = "-124"  # Minimum longitude-North Anatolian
    Params["max_longitude"] = "-112"  # Maximum longitude-North Anatolian

    Params["order_by"] = "time"  # Ordering the data by parameters
    Params["limit_data"] = "20000"  # Maximum number of data

    df = get_EQ_USGS_df(Params)

    file_path = "datasets/SA-1923-2023/" + "SA" + str(start_num) + "_1_newraw" + ".csv"
    df.to_csv(file_path)
    # Second quarter of the year

    start_str = f"""{start_num}-03-02"""
    end_str = f"""{start_num}-06-01"""

    Params["min_date"] = start_str  # Minimum date for reporting the data
    Params["max_date"] = end_str  # Maximum date for reporting the data

    df = get_EQ_USGS_df(Params)

    file_path = "datasets/SA-1923-2023/" + "SA" + str(start_num) + "_2_newraw" + ".csv"
    df.to_csv(file_path)

    # Third quarter of the year

    start_str = f"""{start_num}-06-02"""
    end_str = f"""{start_num}-09-01"""

    Params["min_date"] = start_str  # Minimum date for reporting the data
    Params["max_date"] = end_str  # Maximum date for reporting the data

    df = get_EQ_USGS_df(Params)

    file_path = "datasets/SA-1923-2023/" + "SA" + str(start_num) + "_3_newraw" + ".csv"
    df.to_csv(file_path)

    # Fourth quarter of the year

    start_str = f"""{start_num}-09-02"""
    end_str = f"""{start_num + 1}-01-01"""

    Params["min_date"] = start_str  # Minimum date for reporting the data
    Params["max_date"] = end_str  # Maximum date for reporting the data

    # Getting the dataframe calling the custom-built function
    df = get_EQ_USGS_df(Params)
    # Saving the dataframe as a .csv file
    file_path = "datasets/SA-1923-2023/" + "SA" + str(start_num) + "_4_newraw" + ".csv"

    df.to_csv(file_path)


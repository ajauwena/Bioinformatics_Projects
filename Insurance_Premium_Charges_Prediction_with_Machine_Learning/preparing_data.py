"""
=========================================
Preparing Data for Analysis
=========================================
Authors:
    Abelhard Jauwena, ajauwena@gmail.com
    Olusegun Odumosu, odumosu.segun@gmail.com
Date: April 21, 2023

This script cleans the input data, removes outliers, explores and visualizes the cleaned data, shuffles the data, and outputs it as a .csv

How to run:   python3  preparing_data.py -in insurance_dataset.csv

Output figures:
Outliers_KNN_method.png

Output files:
insurance_dataset_clean.csv
=========================================
"""


# So we're adding stuff from these scripts, in this order: data_cleaner.py, explore_and_visualize_initial_data.py, outlier_detection_KNN_clustering.py, shuffling data (in the data_splitter.py script I think)
import numpy as np
import pandas as pd
import argparse
from numpy import min, max
import matplotlib.pyplot as plt 
from matplotlib import pyplot
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='Preparing the data')
parser.add_argument('--in_file', '-in', default="insurance_dataset.csv", required=False, action="store", dest ='in_file')
args = parser.parse_args()

filename      = args.in_file
insurance_premium_df = pd.read_csv(filename, header=0, encoding="utf8")


# region Cleaning, encoding, and shuffling the data  


# Obtain the column names of the DataFrame.
column_names = insurance_premium_df.columns.values.tolist() # (How to, 2022).
# print(column_names)

# Check for NAs in the DataFrame.
insurance_premium_df.isnull().values.any() # (Check for, n.d.).
# There are no NAs in the DataFrame.

# Convert "bmi" from str to int.
insurance_premium_df['bmi'] = insurance_premium_df['bmi'].astype(float)

# One-hot encode the "region" column.
one_hot = pd.get_dummies(data=insurance_premium_df['region'])

# Combine the one-hot encoded column with the original DataFrame.
insurance_premium_df = pd.concat([insurance_premium_df, one_hot], axis=1)

# Convert all categorical variables to numerical variables.
le_sex = LabelEncoder() # (Denis, 2018).
insurance_premium_df['sex'] = le_sex.fit_transform(insurance_premium_df['sex']) # (Denis, 2018).
le_smoker = LabelEncoder() # (Denis, 2018).
insurance_premium_df['smoker'] = le_smoker.fit_transform(insurance_premium_df['smoker']) # (Denis, 2018).
le_region = LabelEncoder() # (Denis, 2018).
insurance_premium_df['region'] = le_region.fit_transform(insurance_premium_df['region']) # (Denis, 2018).

# Assign 3 bins for "charges"
charges_bins = [0, insurance_premium_df.charges.quantile(0.3333333333), insurance_premium_df.charges.quantile(0.666666), 9999999999] # (Jain, 2020).

# Create classes for "charges."
charges_classes = [1, 2, 3] # (Wisdom, 2019).

# Append the classes for "charges" to the DataFrame.
insurance_premium_df['charge_classes'] = pd.cut(insurance_premium_df.charges, bins=charges_bins, labels=charges_classes) # (Wisdom, 2019).

# Reorder the columns in the DataFrame.
insurance_premium_df = insurance_premium_df.reindex(columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'northeast', 'northwest', 'southeast', 'southwest', 'charges', 'charge_classes'])

# Drop the column "region."
insurance_premium_df.drop('region', inplace=True, axis=1)

# Shuffle the DataFrame.
insurance_premium_df = insurance_premium_df.sample(frac=1, random_state=7).reset_index(drop=True)



# References
# Check for NaN in Pandas DataFrame (examples included). (2021, September 10). Data to Fish. Retrieved March 24, 2023, from https://datatofish.com/check-nan-pandas-dataframe/
# Denis, B. (2018, May 5). Health Care Cost Analysys/Prediction Python. Kaggle. https://www.kaggle.com/code/flagma/health-care-cost-analysys-prediction-python 
# How to Get Column Names in Pandas? (2022, December 1). Board Infinity. Retrieved March 24, 2023, from https://www.boardinfinity.com/blog/how-to-get-column-names-in-pandas/
# Wisdom, G. [Gurukul Wisdom]. (2019, February 22). 27 - Pandas - pandas.cut() Method Explained Clearly [Video]. YouTube. https://www.youtube.com/watch?v=rRTbSH5fOTc&ab_channel=GurukulWisdom


# endregion Cleaning and encoding the data 


#############################
# region ######## Outlier Detection (KNN Clustering) 
#############################

##################
# set font sizes #
##################
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# select charges feature to detect outliers
charges = insurance_premium_df.iloc[:, -2]

# set figure sizes
plt.figure(figsize=(20,10))

# region ## view data before outlier removal 
# plot a scatterplot of charges
# plt subplot as part of matrix of plots
plt.subplot(1,3,1)
plt.scatter(x = charges.index, y = charges.values, s = 20, color = "b")
plt.title("Charges of each data point")
plt.ylabel("Charges ($)")
plt.xlabel("Indices")

# endregion ## view data before outlier removal 



# region ## Perform outlier detection with KNN 
# create the data array
charges_array = charges.values
# print(charges_array.shape)
#reshape array to use with nearest neighbour
reshaped_array = charges_array.reshape(-1, 1)
# print(reshaped_array.shape)

# create and fit model to data
k = 3
nbrs = NearestNeighbors(n_neighbors = k)
nbrs.fit(reshaped_array)

# calculate distances and indexes of k-neighbors for each data point
distances, indexes = nbrs.kneighbors(reshaped_array)

# plot mean of k-distances of each observation
plt.subplot(1,3,2)
plt.plot(distances.mean(axis =1))
plt.axhline(y = 750, color = "r", linestyle = "dashed")
plt.title("Mean of k-distances of each observation")
plt.ylabel("Mean k distance")
plt.xlabel("Indices")
plt.legend(["K-distances", "cutoff"])

# visually determine cutoff of 750
outlier_index = np.where(distances.mean(axis = 1) > 750)
# print("Outlier indices: ", outlier_index)

# extract outlier values
outlier_values = charges.iloc[outlier_index]

# remove outlier values
# print("Data size before outlier removal: ", insurance_premium_df.shape)
insurance_premium_no_outliers = np.delete(insurance_premium_df.values, outlier_index, axis = 0)

# change array to dataframe
clean_dataset_df = pd.DataFrame(insurance_premium_no_outliers)
# print("Data size of dataframe after outlier removal: ", clean_dataset_df.shape)
# endregion ## Perform outlier detection with KNN 



# region ## View data after outlier removal 
# plot data
plt.subplot(1,3,3)
plt.scatter(x = charges.index, y = charges.values, color = "b", s = 20)
# plot outlier values
plt.scatter(x = outlier_index, y = outlier_values, color = "r")
plt.legend(["Inlier", "outlier"])
plt.title("Charges of each data point")
plt.ylabel("Charges ($)")
plt.xlabel("Indices")

plt.tight_layout()
plt.savefig('Outliers_KNN_method.png',dpi=600)
plt.show()
# endregion ## View data after outlier removal 

# remove outlier values
# print("Data size before outlier removal: ", insurance_premium_df.shape)
insurance_premium_no_outliers = np.delete(insurance_premium_df.values, outlier_index, axis = 0)
# change array to dataframe
column_names2 = insurance_premium_df.columns.values.tolist()
clean_dataset_df = pd.DataFrame(data = insurance_premium_no_outliers, columns= column_names2)
# print(insurance_premium_df)
# print(clean_dataset_df)
# print("Data size array after outlier removal: ", insurance_premium_no_outliers.shape)

# Output the resulting DataFrame to a .csv file.
clean_dataset_df.to_csv("insurance_dataset_clean.csv",index=False)


# endregion ####### Outlier Detection (KNN Clustering) 





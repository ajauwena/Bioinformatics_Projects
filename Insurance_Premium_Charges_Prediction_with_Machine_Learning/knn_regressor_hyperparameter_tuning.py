"""
=========================================
Tuning Hyperparameters for KNN Regressor
=========================================
Authors:
    Abelhard Jauwena, ajauwena@gmail.com
Date: April 21, 2023

How to run:   python3 knn_regressor_hyperparameter_tuning.py -in insurance_dataset_clean.csv -k 1 -mk 50

Output figures:
KNN_Regressor_Hyperparameter_Tuning.png
=========================================
"""

# --- IMPORTING MODULES ---

import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error

##################
# set font sizes #
##################
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# --- PARSING ARGUMENTS ---

# Create an argument parser.
parser = argparse.ArgumentParser(description='data quality control')

# Define a command line argument for the data set.
parser.add_argument('-in', '--in_file', action='store', dest='in_file', required=False, default='insurance_dataset_clean.csv', help='The name of the .csv file containing the data set of interest, where the last column is the desired output.')

# Define a command line argument for the number of neighbors. The default is 5.
parser.add_argument('-k', '--num_neighbors', action='store', dest='kk', default=5, required=False, help='The desired number of neighbors.')

# Define a command line argument for the maximum number of neighbors. The default is 50.
parser.add_argument('-mk', '--max_num_neighbors', action='store', dest='max_kk', default=50, required=False, help='The desired maximum number of neighbors, which should not exceed the number of rows in the data set.')

# Handle potential user errors.
try:
    arguments = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# Store the arguments in separate variables.
file_name = arguments.in_file
kk = int(arguments.kk)
max_kk = int(arguments.max_kk)

# --- PROCESSING THE DATA ---

# Read the data set as a pandas DataFrame, treating the first column as the header.
df = pd.read_csv(file_name, header=0)

# Create a new column containing the values from the columns "smoker" and "charge_classes," which will be used to stratify the data.
df['stratify_column'] = df['smoker'].astype(str) + df['charge_classes'].astype(str)

# Obtain the data from the DataFrame.
data = df.values
# "data" is a list containing lists, where each list corresponds to a row in the DataFrame "df."

# Assign the data in all columns except the columns "charges", "charge_classes," and "stratify_column" as input variables.
X = data[:, :-3]

# Assign the data in the column "charges" as the output variable.
y = data[:, -3].astype(int)

# Split the data using a 70:30 train:test split with stratified sampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=7, stratify=df['stratify_column'])

# Print the dimensions of the training and testing sets for both the input and output variables.
print('x_train size:\n', X_train.shape, '\n')
print('x_test size:\n', X_test.shape, '\n')
print('y_train size:\n', y_train.shape, '\n')
print('y_test size:\n', y_test.shape, '\n')

# --- FITTING AND TRANSFORMING THE INPUT VARIABLES ---

# Create a StandardScaler object.
scaler = StandardScaler()

# Compute the means and standard deviations of the input variables in the training set.
scaler.fit(X_train)

# Transform the input variables in both the training and testing sets using their respective means and standard deviations, which were computed above.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# --- DETERMINING THE BEST K VALUE USING R^2 AND RMSE SCORES ---

# Handle a potential error whereby the maximum number of neighbors (k) is greater than the number of rows in the training set for the input variables.
if (max_kk > X_train.shape[0]):
    max_kk = X_train.shape[0]

# Create an empty list "r2" that will contain R^2 scores.
r2 = []

# Create an empty list "rmse" that will contain RMSE scores.
rmse = []

# Compute the R^2 scores for KNN regressors whose k values range from 1 to "max_kk."
for i in range(1, max_kk):
    # Build a KNN regressor.
    knn_regressor = KNeighborsRegressor(n_neighbors=i) # (Singh, 2018).
    # Fit the KNN regressor on the training set.
    knn_regressor.fit(X_train, y_train) # (Singh, 2018).
    # Test the KNN regressor on the testing set.
    y_pred_i = knn_regressor.predict(X_test) # (Singh, 2018).
    # Append the R^2 score for a k value to the list "r2."
    r2.append(r2_score(y_test, y_pred_i)) # (Singh, 2018).
    # Append the RMSE score for a k value to the list "rmse."
    rmse.append(np.sqrt(mean_squared_error(y_test, y_pred_i)))

# Creating a figure with scatterplots for various evaluation measures to determine which value of k is optimal
hp_tuning_fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6.6))
# Plot a scatter plot of the R^2 scores for each k value.
ax[0].plot(range(1, max_kk), r2, color='black', linestyle='solid', marker='o', markerfacecolor='#9dc6d8', markersize=7)
ax[0].set_title('R$^2$ Scores')
ax[0].set_xlabel('Number of Neighbors (k)')
ax[0].set_ylabel('R$^2$ Score')
ax[0].grid()
ax[0].set_xticks([i for i in range(1, 51, 1)])


# Plot a scatter plot of the RMSE scores for each k value.
ax[1].plot(range(1, max_kk), rmse, color='black', linestyle='solid', marker='o', markerfacecolor='#9dc6d8', markersize=7)
ax[1].set_title('RMSE Scores')
ax[1].set_xlabel('Number of Neighbors (k)')
ax[1].set_ylabel('RMSE Score')
ax[1].grid()
ax[1].set_xticks([i for i in range(1, 51, 1)])
hp_tuning_fig.suptitle("Evaluation scores for the KNN regressor with different values of k",fontsize=18, fontweight='bold', y = 0.99)
hp_tuning_fig.tight_layout()
hp_tuning_fig.savefig('KNN_Regressor_Hyperparameter_Tuning.png',dpi=600)

# The best k value as determined by both R^2 and RMSE scores is 6.

# --- REFERENCES ---

# Singh, A. (2018, August 22). KNN algorithm: Introduction to K-Nearest Neighbors Algorithm for Regression. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
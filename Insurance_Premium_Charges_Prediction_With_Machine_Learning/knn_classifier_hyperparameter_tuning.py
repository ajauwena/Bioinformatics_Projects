"""
=========================================
Tuning Hyperparameters for KNN Classifier
=========================================
Authors:
    Abelhard Jauwena, ajauwena@gmail.com
Date: April 21, 2023

How to run:   python3 knn_classifier_hyperparameter_tuning.py -in insurance_dataset_clean.csv -k 5 -mk 50

Output figures:
KNN_Classifier_Hyperparameter_Tuning.png
=========================================
"""

# --- IMPORTING MODULES ---

import pandas as pd
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

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
# "data" is a list containing lists as elements, where each list corresponds to a row in the DataFrame "df."

# Assign the data in all columns except the columns "charges", "charge_classes," and "stratify_column" as input variables.
X = data[:, :-3]

# Assign the data in the column "charge_classes" as the output variable.
y = data[:, -2].astype('int')

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

# Compute the means and standard deviations for the input variables in the training set.
scaler.fit(X_train)

# Transform the input variables in both the training and testing sets using their respective means and standard deviations, which were computed above.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# --- DETERMINING THE BEST K VALUE USING ACCURACY AND F1 SCORE ---

# Handle a potential error whereby the user specifies a "max_kk" value that is greater than the number of rows in the training set for the input variables.
if (max_kk > X_train.shape[0]):
    max_kk = X_train.shape[0]

# Create an empty list "error" that will contain mean prediction errors.
error = []

# Create an empty list "accuracy" that will contain accuracy scores.
accuracy = []

# Create an empty list "f1" that will contain F1 scores.
f1 = []

# Compute the mean prediction errors for KNN classifiers whose k values range from 1 to "max_kk."
for i in range(1, max_kk):
    # Build a KNN classifier.
    knn_classifier = KNeighborsClassifier(n_neighbors=i)
    # Fit the KNN classifier on the training set.
    knn_classifier.fit(X_train, y_train)
    # Test the KNN classifier on the testing set.
    y_pred_i = knn_classifier.predict(X_test)
    # Append the mean prediction error a k value to the list "error."
    error.append(np.mean(y_pred_i != y_test))
    # Append the accuracy score of a k value to the list "accuracy."
    accuracy.append(accuracy_score(y_test, y_pred_i))
    # Append the F1 score of a k value to the list "f1."
    f1.append(f1_score(y_test, y_pred_i, average='weighted'))

# Creating a figure with scatterplots for various evaluation measures to determine which value of k is optimal
hp_tuning_fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
# Plot a scatter plot of the mean prediction errors for each k value.
ax[0].plot(range(1, max_kk), error, color='black', linestyle='solid', marker='o', markerfacecolor='#9dc6d8', markersize=7)
ax[0].set_title('Classification Errors')
ax[0].set_xlabel('Number of Neighbors (k)')
ax[0].set_ylabel('Classification Error')
ax[0].grid()
ax[0].set_xticks([i for i in range(1, 51, 1)])


# Plot a scatter plot of the accuracy scores for each k value.
ax[1].plot(range(1, max_kk), accuracy, color='black', linestyle='solid', marker='o', markerfacecolor='#9dc6d8', markersize=7)
ax[1].set_title('Accuracy Scores')
ax[1].set_xlabel('Number of Neighbors (k)')
ax[1].set_ylabel('Accuracy Score')
ax[1].grid()
ax[1].set_xticks([i for i in range(1, 51, 1)])


# Plot a scatter plot of the F1 scores for each k value.
ax[2].plot(range(1, max_kk), f1, color='black', linestyle='solid', marker='o', markerfacecolor='#9dc6d8', markersize=7)
ax[2].set_title('F1 Scores')
ax[2].set_xlabel('Number of Neighbors (k)')
ax[2].set_ylabel('F1 Score')
ax[2].grid()
ax[2].set_xticks([i for i in range(1, 51, 1)])
hp_tuning_fig.suptitle("Evaluation scores for the KNN classifier with different values of k",fontsize=18, fontweight='bold', y = 0.99)
hp_tuning_fig.tight_layout()
hp_tuning_fig.savefig('KNN_Classifier_Hyperparameter_Tuning.png',dpi=600)

# The best k value as determined by mean prediction error, accuracy score, and F1 score is 5.
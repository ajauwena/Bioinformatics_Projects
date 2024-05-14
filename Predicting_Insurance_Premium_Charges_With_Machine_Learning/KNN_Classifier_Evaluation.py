"""
=========================================
Evaluating KNN Classifier Performance
=========================================
Authors:
    Abelhard Jauwena, ajauwena@gmail.com
Date: April 21, 2023

How to run:   python3 knn_classifier_evaluation.py -in insurance_dataset_clean.csv -k 5 -mk 50

Output figures:
KNN_Classifier_Feature_Importances.png
KNN_Classifier_Learning_Curves.png
KNN_Classifier_Confusion_Matrices.png
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
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import csv

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
X_default = data[:, :-3]

# Assign the data in the column "charge_classes" as the output variable.
y_default = data[:, -2].astype('int')

# Split the data using a 70:30 train:test split with stratified sampling.
X_train_default, X_test_default, y_train_default, y_test_default = train_test_split(X_default, y_default, train_size=0.70, random_state=7, stratify=df['stratify_column'])

# Print the dimensions of the training and testing sets for both the input and output variables.
print('Default x train size:\n', X_train_default.shape, '\n')
print('Default x test size:\n', X_test_default.shape, '\n')
print('Default y train size:\n', y_train_default.shape, '\n')
print('Default y test size:\n', y_test_default.shape, '\n')

# --- FITTING AND TRANSFORMING THE INPUT VARIABLES ---

# Create a StandardScaler object.
scaler = StandardScaler()

# Compute the means and standard deviations for the input variables in the training set.
scaler.fit(X_train_default)

# Transform the input variables in both the training and testing sets using their respective means and standard deviations, which were computed above.
X_train_default = scaler.transform(X_train_default)
X_test_default = scaler.transform(X_test_default)

# --- TESTING THE DEFAULT KNN CLASSIFIER ---

# Build a default KNN classifier using k = 5.
knn_classifier_default = KNeighborsClassifier(n_neighbors=5)

# Fit the default KNN classifier on the training set.
knn_classifier_default.fit(X_train_default, y_train_default)

# Test the default KNN classifier on the testing set.
y_pred_default = knn_classifier_default.predict(X_test_default)

# Obtain its accuracy score.
accuracy_default = accuracy_score(y_test_default, y_pred_default)

# View the accuracy score of the default KNN classifier.
print('The accuracy score for the default KNN classifier is: ', '%.4f' % (accuracy_default))

# Obtain its weighted F1 score.
f1_default = f1_score(y_test_default, y_pred_default, average='weighted')

# View the weighted F1 score of the default KNN classifier.
print('The F1 score for the default KNN classifier is: ', f1_default)

# --- PLOTTING LEARNING CURVES FOR THE DEFAULT KNN CLASSIFIER ---

# Define the training sizes to use for plotting the learning curve.
train_sizes_default = np.linspace(0.1, 1.0, 10)

# Compute the learning curve based on accuracy.
train_sizes_default, train_scores_default, test_scores_default = learning_curve(estimator=knn_classifier_default,
                                                                                X=X_default,
                                                                                y=y_default,
                                                                                train_sizes=train_sizes_default,
                                                                                cv=10,
                                                                                scoring='accuracy')

# Compute the means and standard deviations for the training scores based on accuracy.
train_scores_mean_default = np.mean(train_scores_default, axis=1)
train_scores_std_default = np.std(train_scores_default, axis=1)

# Compute the means and standard deviations for the testing scores based on accuracy.
test_scores_mean_default = np.mean(test_scores_default, axis=1)
test_scores_std_default = np.std(test_scores_default, axis=1)

# Plot the learning curve based on accuracy.
learning_curves, ax_lc = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
ax_lc[0].plot(train_sizes_default, train_scores_mean_default, 'o-', color='r', label='Training Accuracy Score')
ax_lc[0].fill_between(train_sizes_default, train_scores_mean_default - train_scores_std_default, train_scores_mean_default + train_scores_std_default, alpha=0.1, color='r')
ax_lc[0].plot(train_sizes_default, test_scores_mean_default, 'o-', color='g', label='Validation Accuracy Score')
ax_lc[0].fill_between(train_sizes_default, test_scores_mean_default - test_scores_std_default, test_scores_mean_default + test_scores_std_default, alpha=0.1, color='g')
ax_lc[0].grid()
ax_lc[0].set_xlabel('Training Set Size')
ax_lc[0].set_ylabel('Accuracy Score')
ax_lc[0].set_title('Default KNN Classifier')
ax_lc[0].legend(loc='best')

# The optimal training set size based on the accuracy score of the default model is around 1,000, which is close enough to the size of the training set that I used (931). However, adding more training data would benefit the default model, as its accuracy scores for both the training and validation sets continue to increase with increasing training set sizes.

# --- SELECTING THE BEST FEATURES ---

# Fit a KNN classifier on the input and output variables.
knn_classifier_feature_selection = KNeighborsClassifier().fit(X_default, y_default)

# Conduct permutation feature importance for each input variable. Repeat this process 10 times, then obtain the mean importance value for each feature.
importance_values = permutation_importance(knn_classifier_feature_selection, X_default, y_default, n_repeats=10, random_state=23).importances_mean

# Obtain only the names of the input variables in an array.
feature_names = df.columns[:-1][:-2]

# Create an empty dictionary that will store feature names and their respective importance values.
feature_importances = {}

# Loop through the feature names and their corresponding importance values.
for i, j in zip(feature_names, importance_values):
    # Add the feature name as the key and its corresponding importance value as the value to the dictionary.
    feature_importances[i] = j

# Create a DataFrame from the dictionary created above.
df_feature_importances = pd.DataFrame.from_dict(feature_importances, orient='index').rename(columns={0:'Importance'}).sort_values(by='Importance', ascending=False)

# Print the DataFrame.
print(df_feature_importances)

# Plot the feature importances.
feature_importance_fig, ax_fs = plt.subplots()
ax_fs.bar(list(df_feature_importances.index), df_feature_importances.loc[:, 'Importance'], color ="#9dc6d8")
ax_fs.set_xlabel('Feature')
ax_fs.set_xticks(range(len(df_feature_importances.index)))
ax_fs.set_xticklabels(list(df_feature_importances.index),rotation=45)
ax_fs.set_ylabel('Importance')
ax_fs.set_title('KNN classifier feature importances')
feature_importance_fig.tight_layout()
feature_importance_fig.savefig('KNN_Classifier_Feature_Importances.png',dpi=600)

# Choosing the five best features.
top_5_features = df_feature_importances.index[:5]
X_selected = df[top_5_features].values

# --- OBTAINING THE INPUT AND OUTPUT VARIABLES FOR THE FINAL KNN CLASSIFIER ---

# Assign the data in the columns "age," "bmi," "children," "smoker," and "northeast" as the selected input variables.
X_final = X_selected[:, :]

# Assign the data in the column "charge_classes" as the output variable.
y_final = data[:, -2].astype('int')

# Split the data using a 70:30 train:test split with stratified sampling.
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, train_size=0.70, random_state=7, stratify=df['stratify_column'])

# Print the dimensions of the training and testing sets for both the input and output variables.
print('Final x train size:\n', X_train_final.shape, '\n')
print('Final x test size:\n', X_test_final.shape, '\n')
print('Final y train size:\n', y_train_final.shape, '\n')
print('Final y test size:\n', y_test_final.shape, '\n')

# --- FITTING AND TRANSFORMING THE INPUT VARIABLES FOR THE FINAL KNN CLASSIFIER ---

# Create a StandardScaler object.
scaler = StandardScaler()

# Compute the means and standard deviations of the input variables in the training set.
scaler.fit(X_train_final)

# Transform the input variables in both the training and testing sets using their respective means and standard deviations, which were computed above.
X_train_final = scaler.transform(X_train_final)
X_test_final = scaler.transform(X_test_final)

# --- TESTING THE FINAL KNN CLASSIFIER ---

# Build the final KNN classifier using k = 5.
knn_classifier_final = KNeighborsClassifier(n_neighbors=5)

# Fit the final KNN classifier on the training set.
knn_classifier_final.fit(X_train_final, y_train_final)

# Test the final KNN classifier on the testing set.
y_pred_final = knn_classifier_final.predict(X_test_final).astype(int)

# Obtain its accuracy score.
accuracy_final = accuracy_score(y_test_final, y_pred_final)

# View the accuracy score of the final KNN classifier.
print('The accuracy score for the final KNN classifier is: ', '%.4f' % (accuracy_final))

# Obtain its weighted F1 score.
f1_final = f1_score(y_test_final, y_pred_final, average='weighted')

# View the weighted F1 score of the final KNN classifier.
print('The F1 score for the final KNN classifier is: ', f1_final)

# --- PLOTTING LEARNING CURVES FOR THE FINAL KNN CLASSIFIER ---

# Define the training sizes to use for plotting the learning curve.
train_sizes_final = np.linspace(0.1, 1.0, 10)

# Compute the learning curve based on accuracy.
train_sizes_final, train_scores_final, test_scores_final = learning_curve(estimator=knn_classifier_final,
                                                                          X=X_final,
                                                                          y=y_default,
                                                                          train_sizes=train_sizes_final,
                                                                          cv=10,
                                                                          scoring='accuracy')

# Compute the means and standard deviations for the training scores based on accuracy.
train_scores_mean_final = np.mean(train_scores_final, axis=1)
train_scores_std_final = np.std(train_scores_final, axis=1)

# Compute the means and standard deviations for the testing scores based on accuracy.
test_scores_mean_final = np.mean(test_scores_final, axis=1)
test_scores_std_final = np.std(test_scores_final, axis=1)

# Plot the learning curve based on accuracy.
# plt.figure(figsize=(8, 6))
ax_lc[1].plot(train_sizes_final, train_scores_mean_final, 'o-', color='r', label='Training Accuracy Score')
ax_lc[1].fill_between(train_sizes_final, train_scores_mean_final - train_scores_std_final, train_scores_mean_final + train_scores_std_final, alpha=0.1, color='r')
ax_lc[1].plot(train_sizes_final, test_scores_mean_final, 'o-', color='g', label='Validation Accuracy Score')
ax_lc[1].fill_between(train_sizes_final, test_scores_mean_final - test_scores_std_final, test_scores_mean_final + test_scores_std_final, alpha=0.1, color='g')
ax_lc[1].grid()
ax_lc[1].set_xlabel('Training Set Size')
ax_lc[1].set_ylabel('Accuracy Score')
ax_lc[1].set_title('Final KNN Classifier')
ax_lc[1].legend(loc='best')
learning_curves.suptitle('Learning curves for the KNN classifiers', fontsize=18, fontweight='bold', y=0.98)
learning_curves.tight_layout()
learning_curves.savefig('KNN_Classifier_Learning_Curves.png',dpi=600)

# The optimal training set size based on the accuracy score of the final model is around 1,000, which is close enough to the size of the training set that I used (931). However, adding more training data would benefit the final model, as its accuracy scores for both the training and validation sets continue to increase with increasing training set sizes.

# --- PLOTTING CONFUSION MATRICES FOR BOTH THE DEFAULT AND FINAL KNN CLASSIFIERS ---

# Plot the template for the confusion matrix.
confusion_matrices, ax_cm = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

# Plot the confusion matrix for the default KNN classifier.
#ax_cm[0].set_title('Confusion matrix for the default KNN classifier')
confusion_matrix_default = confusion_matrix(y_test_default, y_pred_default)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_default).plot(ax=ax_cm[0])
ax_cm[0].set_title('Default KNN Classifier')
ax_cm[0].xaxis.set_ticklabels(['Low', 'Medium', 'High']); ax_cm[0].yaxis.set_ticklabels(['Low', 'Medium', 'High'])

# Plot the confusion matrix for the final KNN classifier.
#ax_cm[1].set_title('Confusion matrix for the final KNN classifier')
confusion_matrix_final = confusion_matrix(y_test_final, y_pred_final)
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_final).plot(ax=ax_cm[1])
ax_cm[1].set_title('Final KNN Classifier')
ax_cm[1].xaxis.set_ticklabels(['Low', 'Medium', 'High']); ax_cm[1].yaxis.set_ticklabels(['Low', 'Medium', 'High'])

# Write the title for the confusion matrix.
confusion_matrices.suptitle('Confusion matrices for the KNN classifiers', fontsize=18, fontweight='bold', y=0.93)
confusion_matrices.savefig('KNN_Classifier_Confusion_Matrices.png',dpi=600)

# --- WRITING THE EVALUATION MEASURES TO A .CSV FILE ---
# specify the file name and new row data
filename = "Classifier_eval_scores.csv"
classifier = 'KNN'
new_row = [classifier, accuracy_default, accuracy_final, f1_default, f1_final]

# open the file in append mode
with open(filename, mode="a", newline="") as csvfile:

    # create a writer object
    writer = csv.writer(csvfile)

    # write the new row to the file
    writer.writerow(new_row)
#knn_precision = precision_score(y_test_final)

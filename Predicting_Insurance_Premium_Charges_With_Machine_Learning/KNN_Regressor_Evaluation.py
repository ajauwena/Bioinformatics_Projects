"""
=========================================
Evaluating KNN Regressor Performance
=========================================
Authors:
    Abelhard Jauwena, ajauwena@gmail.com
Date: April 21, 2023

How to run:   python3 knn_regressor_evaluation.py -in insurance_dataset_clean.csv -k 6 -mk 50

Output figures:
KNN_Regressor_Feature_Importances.png
KNN_Regressor_Learning_Curves.png
KNN_Regressor_Confusion_Matrices.png
=========================================
"""

# --- IMPORTING MODULES ---

import pandas as pd
import numpy as np
import argparse
import sys
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

# --- SETTING FONT SIZES ---

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

# --- OBTAINING THE INPUT AND OUTPUT VARIABLES FOR THE DEFAULT KNN REGRESSOR ---

# Assign the data in all columns except the columns "charges," "charge_classes," and "stratify_column" as input variables.
X_default = data[:, :-3]

# Assign the data in the column "charges" as the output variable.
y_default = data[:, -3].astype(int)

# Split the data using a 70:30 train:test split with stratified sampling.
X_train_default, X_test_default, y_train_default, y_test_default = train_test_split(X_default, y_default, train_size=0.70, random_state=7, stratify=df['stratify_column'])

# Print the dimensions of the training and testing sets for both the input and output variables.
print('Default x train size:\n', X_train_default.shape, '\n')
print('Default x test size:\n', X_test_default.shape, '\n')
print('Default y train size:\n', y_train_default.shape, '\n')
print('Default y test size:\n', y_test_default.shape, '\n')

# --- FITTING AND TRANSFORMING THE INPUT VARIABLES FOR THE DEFAULT KNN REGRESSOR ---

# Create a StandardScaler object.
scaler = StandardScaler()

# Compute the means and standard deviations of the input variables in the training set.
scaler.fit(X_train_default)

# Transform the input variables in both the training and testing sets using their respective means and standard deviations, which were computed above.
X_train_default = scaler.transform(X_train_default)
X_test_default = scaler.transform(X_test_default)

# --- EVALUATING THE DEFAULT KNN REGRESSOR ---

# Build the default KNN regressor using k = 6.
knn_regressor_default = KNeighborsRegressor(n_neighbors=6)

# Fit the default KNN regressor on the training set.
knn_regressor_default.fit(X_train_default, y_train_default)

# Test the default KNN regressor on the testing set.
y_pred_default = knn_regressor_default.predict(X_test_default).astype(int)

# Obtain its R^2 score.
r2_default = r2_score(y_test_default, y_pred_default)

# View the R^2 score of the default KNN regressor.
print('The R^2 score for the default KNN regressor is: ', r2_default)
# The R^2 score of the default KNN regressor is ~0.757.

# Obtain its RMSE score.
rmse_default = np.sqrt(mean_squared_error(y_test_default, y_pred_default))

# View the RMSE score of the default KNN regressor.
print('The RMSE score for the default KNN regressor is: ', rmse_default)
# The RMSE score of the default KNN regressor is ~5698.869.

# --- PLOTTING A SCATTER PLOT FOR THE DEFAULT KNN REGRESSOR ---

scatterplot_fig, ax_scp = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
# Plot a scatter plot comparing the actual vs. predicted charges.
ax_scp[0].scatter(y_test_default, y_pred_default, color='#7dd0b6', s=15)

# Calculate the line of best fit.
m_default, b_default = np.polyfit(y_test_default.astype(float), y_pred_default, 1)

# Plot the line of best fit.
ax_scp[0].plot(y_test_default, m_default * y_test_default + b_default, color='red', label='Line of Best Fit')
ax_scp[0].text(600,42000,f'Line of best fit: {m_default:.2f}x + {b_default:.2f}')
print('The slope of the line of best fit is: ', m_default, '\n')

# Plot a line showing where the line of best fit would be if the KNN regressor predicted perfectly.
m_default, b_default = np.polyfit(y_test_default.astype(float), y_test_default.astype(float), 1)
ax_scp[0].plot(y_test_default, m_default * y_test_default + b_default, color='black')
print('The slope of the ideal line of best fit is: ', m_default, '\n')

# Calculating and plotting the Concordance Correlation Coefficient (CCC)
y_true_mean = np.mean(y_test_default)
y_pred_mean = np.mean(y_pred_default)
s_xy = np.sum((y_test_default - y_true_mean) * (y_pred_default - y_pred_mean)) / (len(y_test_default) - 1)
s_x2 = np.sum((y_test_default - y_true_mean) ** 2) / (len(y_test_default) - 1)
s_y2 = np.sum((y_pred_default - y_pred_mean) ** 2) / (len(y_test_default) - 1)
r = s_xy / np.sqrt(s_x2 * s_y2)
ccc = 2 * r * np.std(y_test_default) * np.std(y_pred_default) / (np.var(y_test_default) + np.var(y_pred_default) + (y_true_mean - y_pred_mean) ** 2)
ax_scp[0].text(x=600,y=40000,s=f'CCC = {ccc:.4f}')

# Add labels and legend.
ax_scp[0].set_xlabel('Ground Truth Value')
ax_scp[0].set_ylabel('Predicted Value')
r2_plot_default = r2_score(y_test_default, y_pred_default)
ax_scp[0].text(x=600, y=38000, s=f'R$^2$ = {r2_plot_default:.2f}')
ax_scp[0].set_title('Default KNN Regressor')
ax_scp[0].legend()


# --- PLOTTING LEARNING CURVES FOR THE DEFAULT KNN REGRESSOR ---

# Define the training sizes to use for plotting the learning curve.
train_sizes_default = np.linspace(0.1, 1.0, 10)

# Compute the learning curve based on R^2 score.
train_sizes_default, train_scores_default, test_scores_default = learning_curve(estimator=knn_regressor_default,
                                                                                X=X_default,
                                                                                y=y_default,
                                                                                train_sizes=train_sizes_default,
                                                                                cv=10,
                                                                                scoring='r2')

# Compute the means and standard deviations for the training scores based on R^2 score.
train_scores_mean_default = np.mean(train_scores_default, axis=1)
train_scores_std_default = np.std(train_scores_default, axis=1)

# Compute the means and standard deviations for the testing scores based on R^2 score.
test_scores_mean_default = np.mean(test_scores_default, axis=1)
test_scores_std_default = np.std(test_scores_default, axis=1)

# Plot the learning curve based on R^2 scores.
learning_curves, ax_lc = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
ax_lc[0].plot(train_sizes_default, train_scores_mean_default, 'o-', color='r', label='Training R$^2$ Score')
ax_lc[0].fill_between(train_sizes_default, train_scores_mean_default - train_scores_std_default, train_scores_mean_default + train_scores_std_default, alpha=0.1, color='r')
ax_lc[0].plot(train_sizes_default, test_scores_mean_default, 'o-', color='g', label='Validation R$^2$ Score')
ax_lc[0].fill_between(train_sizes_default, test_scores_mean_default - test_scores_std_default, test_scores_mean_default + test_scores_std_default, alpha=0.1, color='g')
ax_lc[0].grid()
ax_lc[0].set_xlabel('Training Set Size')
ax_lc[0].set_ylabel('R$^2$ Score')
ax_lc[0].set_title('Default KNN Regressor')
ax_lc[0].legend(loc='best')


# The optimal training set size based on the R^2 score of the default model is around 1,000, which is close enough to the size of the training set that I used (931). However, adding more training data would benefit the default model, as its R^2 scores for both the training and validation sets continue to increase with increasing training set sizes.

# --- SELECTING THE BEST FEATURES ---

# Fit a KNN regressor to the entire input and output variables.
knn_regressor_feature_selection = KNeighborsRegressor().fit(X_default, y_default)

# Conduct permutation feature importance for each input variable. Repeat this process 10 times, then obtain the mean importance value for each feature.
importance_values = permutation_importance(knn_regressor_feature_selection, X_default, y_default, n_repeats=10, random_state=23).importances_mean

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

# Plot the feature importances.
feature_importance_fig, ax_fs = plt.subplots()
ax_fs.bar(list(df_feature_importances.index), df_feature_importances.loc[:, 'Importance'], color ="#9dc6d8")
ax_fs.set_xlabel('Feature')
ax_fs.set_xticks(range(len(df_feature_importances.index)))
ax_fs.set_xticklabels(list(df_feature_importances.index),rotation=45)
ax_fs.set_ylabel('Importance')
ax_fs.set_title('KNN regressor feature importances')
feature_importance_fig.tight_layout()
feature_importance_fig.savefig('KNN_Regressor_Feature_Importances.png',dpi=600)

# Choosing the five best features.
top_5_features = df_feature_importances.index[:5]
X_selected = df[top_5_features].values

# --- OBTAINING THE INPUT AND OUTPUT VARIABLES FOR THE FINAL KNN REGRESSOR ---

# Assign the data in the columns "age," "bmi," "smoker," "children," and "sex" as the selected input variables.
X_final = X_selected

# Assign the data in the column "charges" as the output variable.
y_final = data[:, -3].astype(int)

# Split the data using a 70:30 train:test split with stratified sampling.
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final, y_final, train_size=0.70, random_state=7, stratify=df['stratify_column'])

# Print the dimensions of the training and testing sets for both the input and output variables.
print('Final x train size:\n', X_train_final.shape, '\n')
print('Final x test size:\n', X_test_final.shape, '\n')
print('Final y train size:\n', y_train_final.shape, '\n')
print('Final y test size:\n', y_test_final.shape, '\n')

# --- FITTING AND TRANSFORMING THE INPUT VARIABLES FOR THE FINAL KNN REGRESSOR ---

# Create a StandardScaler object.
scaler = StandardScaler()

# Compute the means and standard deviations of the input variables in the training set.
scaler.fit(X_train_final)

# Transform the input variables in both the training and testing sets using their respective means and standard deviations, which were computed above.
X_train_final = scaler.transform(X_train_final)
X_test_final = scaler.transform(X_test_final)

# --- TESTING THE FINAL KNN REGRESSOR ---

# Build the final KNN regressor using k = 6.
knn_regressor_final = KNeighborsRegressor(n_neighbors=6)

# Fit the final KNN regressor on the training set.
knn_regressor_final.fit(X_train_final, y_train_final)

# Test the final KNN regressor on the testing set.
y_pred_final = knn_regressor_final.predict(X_test_final).astype(int)

# Obtain its R^2 score.
r2_final = r2_score(y_test_final, y_pred_final)

# View the R^2 score of the final KNN regressor.
print('The R^2 score for the final KNN regressor is: ', r2_final)
# The R^2 score of the final KNN regressor is ~0.810.

# Obtain its RMSE score.
rmse_final = np.sqrt(mean_squared_error(y_test_final, y_pred_final))

# View the RMSE score of the final KNN regressor.
print('The RMSE score for the final KNN regressor is: ', rmse_final)
# The RMSE score of the final KNN regressor is ~5040.218.

# --- PLOTTING A SCATTER PLOT OF ACTUAL VS. PREDICTED OUTPUTS FOR THE FINAL KNN REGRESSOR ---

# Plot a scatter plot comparing the actual vs. predicted charges.
ax_scp[1].scatter(y_test_final, y_pred_final, color='#7dd0b6', s=15)

# Calculate the line of best fit.
m_final, b_final = np.polyfit(y_test_final.astype(float), y_pred_final, 1)

# Plot the line of best fit.
ax_scp[1].plot(y_test_final, m_final * y_test_final + b_final, color='red', label='Line of Best Fit')
ax_scp[1].text(600,42000,f'Line of best fit: {m_final:.2f}x + {b_final:.2f}')
print('The slope of the line of best fit is: ', m_final, '\n')

# Plot a line showing where the line of best fit would be if the KNN regressor predicted perfectly.
m_final, b_final = np.polyfit(y_test_final.astype(float), y_test_final.astype(float), 1)
ax_scp[1].plot(y_test_final, m_final * y_test_final + b_final, color='black')
print('The slope of the ideal line of best fit is: ', m_final, '\n')


# Calculating and plotting the Concordance Correlation Coefficient (CCC)
y_true_mean = np.mean(y_test_final)
y_pred_mean = np.mean(y_pred_final)
s_xy = np.sum((y_test_final - y_true_mean) * (y_pred_final - y_pred_mean)) / (len(y_test_final) - 1)
s_x2 = np.sum((y_test_final - y_true_mean) ** 2) / (len(y_test_final) - 1)
s_y2 = np.sum((y_pred_final - y_pred_mean) ** 2) / (len(y_test_final) - 1)
r = s_xy / np.sqrt(s_x2 * s_y2)
ccc = 2 * r * np.std(y_test_final) * np.std(y_pred_final) / (np.var(y_test_final) + np.var(y_pred_final) + (y_true_mean - y_pred_mean) ** 2)
ax_scp[1].text(x=600,y=40000,s=f'CCC = {ccc:.4f}')

# Add labels and legend.
ax_scp[1].set_xlabel('Ground Truth Value')
ax_scp[1].set_ylabel('Predicted Value')
r2_plot_final = r2_score(y_test_final, y_pred_final)
ax_scp[1].text(x=600, y=38000, s=f'R$^2$ = {r2_plot_final:.2f}')

ax_scp[1].set_title('Final KNN Regressor')
ax_scp[1].legend()
scatterplot_fig.suptitle('KNN Regressor scatterplots of charge predictions of test data', fontsize=18, fontweight='bold', y=0.98)
scatterplot_fig.tight_layout()
scatterplot_fig.savefig('KNN_Regressor_Scatterplots.png',dpi=600)

# --- PLOTTING LEARNING CURVES USING R^2 SCORE FOR THE FINAL KNN REGRESSOR ---

# Define the training sizes to use for plotting the learning curve.
train_sizes_final = np.linspace(0.1, 1.0, 10)

# Compute the learning curve based on R^2 score.
train_sizes_final, train_scores_final, test_scores_final = learning_curve(estimator=knn_regressor_final,
                                                                          X=X_final,
                                                                          y=y_final,
                                                                          train_sizes=train_sizes_final,
                                                                          cv=10,
                                                                          scoring='r2')

# Compute the means and standard deviations for the training scores based on R^2 score.
train_scores_mean_final = np.mean(train_scores_final, axis=1)
train_scores_std_final = np.std(train_scores_final, axis=1)

# Compute the means and standard deviations for the testing scores based on R^2 score.
test_scores_mean_final = np.mean(test_scores_final, axis=1)
test_scores_std_final = np.std(test_scores_final, axis=1)

# Plot the learning curve based on R^2 score.
ax_lc[1].plot(train_sizes_final, train_scores_mean_final, 'o-', color='r', label='Training R$^2$ Score')
ax_lc[1].fill_between(train_sizes_final, train_scores_mean_final - train_scores_std_final, train_scores_mean_final + train_scores_std_final, alpha=0.1, color='r')
ax_lc[1].plot(train_sizes_final, test_scores_mean_final, 'o-', color='g', label='Validation R$^2$ Score')
ax_lc[1].fill_between(train_sizes_final, test_scores_mean_final - test_scores_std_final, test_scores_mean_final + test_scores_std_final, alpha=0.1, color='g')
ax_lc[1].grid()
ax_lc[1].set_xlabel('Training Set Size')
ax_lc[1].set_ylabel('R$^2$ Score')
ax_lc[1].set_title('Final KNN Regressor')
ax_lc[1].legend(loc='best')
learning_curves.suptitle('Learning curves for the KNN regressors', fontsize=18, fontweight='bold', y=0.98)
learning_curves.tight_layout()
learning_curves.savefig('KNN_Regressor_Learning_Curves.png',dpi=600)

# The optimal training set size based on the R^2 score of the final model is around 1,000, which is close enough to the size of the training set that I used (931). However, adding more training data would benefit the final model, as its R^2 scores for both the training and validation sets continue to increase with increasing training set sizes.

# --- WRITING THE EVALUATION MEASURES TO A .CSV FILE ---

# Obtain the RMSE and R^2 scores for the final model.
rmse_default_to_csv = np.sqrt(mean_squared_error(y_test_default, y_pred_default))
r2_default_to_csv = r2_score(y_test_default, y_pred_default)

# Obtain the RMSE and R^2 scores for the final model.
rmse_final_to_csv = np.sqrt(mean_squared_error(y_test_final, y_pred_final))
r2_final_to_csv = r2_score(y_test_final, y_pred_final)

# specify the file name and new row data
filename = "Regressor_eval_scores.csv"
classifier = 'KNN'
new_row = [classifier, r2_default_to_csv, r2_final_to_csv, rmse_default_to_csv, rmse_final_to_csv]

# open the file in append mode
with open(filename, mode="a", newline="") as csvfile:

    # create a writer object
    writer = csv.writer(csvfile)

    # write the new row to the file
    writer.writerow(new_row)

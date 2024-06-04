"""
===================
Regression with SVM (SVR)
===================
Author: Nishita Sharif, nishita.sharif@gmail.com
Date: April 21, 2023

How to run:   python3  SVM_regressor.py insurance_dataset_clean.csv

Output figures:
SVR_LearningCurves.png
SVR_Feature_Importances.png 
SVR_Scatterplots.png 
=================
"""

# Importing modules
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy.stats import uniform
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



# Defining function that will split and scale data
def split_and_scale_data(X, y):
    # Split the data using a 70:30 train:test split with stratified sampling.
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=7, stratify=insurance_premium_df['stratify_column'])

    # Initiating a StandardScaler object and fitting the scaler to the training data
    scaler = StandardScaler().fit(X_train)

    # Scaling the training and testing input (predictor) data 
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return(X_train_scaled,X_test_scaled,y_train, y_test)
    
# Defining function that will plot learning curve; can be used to combine learning curves into one figure
def plot_learning_curve(estimator, X, y, train_sizes, cv, title, axis):
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring= 'r2')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    axis.set_title(title)
    axis.set_xlabel("Training examples")
    axis.set_ylabel("R$^2$ Score")
    axis.grid()
    axis.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axis.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axis.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axis.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axis.legend(loc="best")
    
# Defining function that calculates the Concordance Correlation Coefficient (CCC)
def concordance_correlation_coefficient(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    s_xy = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean)) / (len(y_true) - 1)
    s_x2 = np.sum((y_true - y_true_mean) ** 2) / (len(y_true) - 1)
    s_y2 = np.sum((y_pred - y_pred_mean) ** 2) / (len(y_true) - 1)
    r = s_xy / np.sqrt(s_x2 * s_y2)
    ccc = 2 * r * np.std(y_true) * np.std(y_pred) / (np.var(y_true) + np.var(y_pred) + (y_true_mean - y_pred_mean) ** 2)
    return ccc

# Defining function that will plot scatterplots of ground truth vs predicted values; can be used to combine scatterplots into one figure
def plot_scatterplot(y_test,y_pred,title,axis):
    axis.scatter(y_test, y_pred, color='#7dd0b6', s=15)

    # Calculate and plot the best-fit line
    m, b = np.polyfit(y_test.astype(float), y_pred, 1)
    axis.plot(y_test, m*y_test + b, color='red', label='Best fit')
    axis.text(600,42000,f'Line of best fit: {m:.2f}x + {b:.2f}')

    # Black line that shows where the best fit line would be if the model predicter perfectly
    m, b = np.polyfit(y_test.astype(float), y_test.astype(float), 1)
    axis.plot(y_test, m*y_test + b, color='black')

    # Add labels and legend
    axis.set_xlabel('Ground Truth Values')
    axis.set_ylabel('Predicted Values')
    svr_r2=r2_score(y_test,y_pred)
    ccc = concordance_correlation_coefficient(y_test, y_pred)
    axis.text(600,38000,f'R$^2$ = {svr_r2:.4f}')
    axis.text(600,40000,f'CCC = {ccc:.4f}')
    axis.set_title(title)
    axis.legend()



# ----------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='The path to the file to process')
args = parser.parse_args()
clean_dataset      = args.input_file
# Read the cleaned .csv file as a DataFrame
insurance_premium_df = pd.read_csv(clean_dataset, header=0, encoding="utf8")

# Define the columns to use for stratification
insurance_premium_df["stratify_column"] = insurance_premium_df["smoker"].astype(str) + insurance_premium_df["charge_classes"].astype(str)

# Assigning the values in the dataframe columns to a variable
data = insurance_premium_df.values

# Assign the data in all columns except the columns "charges", "charge_classes" and "stratify_column" as input variables
X = data[:, :-3]

# Assign the data in the column "charges" as the output variable
y = data[:, -3].astype(np.float64)

# Splitting and scaling data, fitting, then predicting on test data
X_train_scaled, X_test_scaled, y_train, y_test=split_and_scale_data(X,y)
default_svr_pred = SVR().fit(X_train_scaled, y_train).predict(X_test_scaled)

# Plotting scatterplot of ground truth data vs predictions made by default model
scatterplot_fig, ax_scp = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
plot_scatterplot(y_test,default_svr_pred,'Default SVR Model',ax_scp[0])

# Calculating and storing the RMSE for the default model
default_svr_rmse=np.sqrt(mean_squared_error(y_test,default_svr_pred))
default_svr_r2=r2_score(y_test,default_svr_pred)

# Creating learning curves figure, which will have 3 plots
learning_curves_fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,7))
learning_curves_fig.subplots_adjust(hspace=0.3, wspace=0.13)

# Plotting the learning curve for the default (scaled) data
plot_learning_curve(SVR(), X_train_scaled, y_train, np.linspace(0.1, 1.0, 10), 10, "Default", ax[0])



# region Feature Selection
svr = SVR().fit(X_train_scaled, y_train)
importance = permutation_importance(svr, X_train_scaled, y_train, n_repeats=10, random_state=23).importances_mean
feature_names = insurance_premium_df.columns[:-1][:-2]

# Use the zip() function to create a dictionary with feature names and their respective importance values
feats = {}
for feature, importance in zip(feature_names, importance):
    feats[feature] = importance #add the name/value pair 

# Create a dataframe using the tuple created above, then sort in descending order
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'}).sort_values(by='Importance', ascending=False)

# View feature importance
print(importances)

# plot feature importance
feature_importance_fig, ax_fs = plt.subplots()
ax_fs.bar(list(importances.index), importances.loc[:,'Importance'], color ="#9dc6d8")
ax_fs.set_xlabel('Features')
ax_fs.set_xticks(range(len(importances.index)))
ax_fs.set_xticklabels(list(importances.index), rotation=45)
ax_fs.set_ylabel('Importance')
ax_fs.set_title("SVR Feature Importance")
feature_importance_fig.tight_layout()
feature_importance_fig.savefig("SVR_Feature_Importances.png",dpi=600)

top_5_features = importances.index[:5]
X_selected_features = insurance_premium_df[top_5_features].values

# Split the data with selected features using a 70:30 train:test split with stratified sampling.
X_train_scaled,X_test_scaled,y_train, y_test=split_and_scale_data(X_selected_features,y)

# Plot the learning curve after feature selection
plot_learning_curve(SVR(), X_train_scaled, y_train, np.linspace(0.1, 1.0, 10), 10, "After Feature Selection", ax[1])

# endregion Feature Selection







# region Tuning hyperparameters

param_space = {
    'C': uniform(0,200),
    # 'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 4],
    'coef0': [0, 1, 2],
    'epsilon': [0.1, 0.2, 0.3],
    'shrinking': [True, False],
    'tol': [0.001, 0.0001]
}

# Create the randomized search object
random_search = RandomizedSearchCV(
    estimator=SVR(),
    param_distributions=param_space,
    n_iter=100,
    cv=10,
    random_state=23,
    n_jobs=-1
)


# Fit the randomized search object to the data
random_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters
print("\nBest hyperparameters for SVR:")
for key, value in random_search.best_params_.items():
    print(f'{key}: {value}')


svr_model = SVR(C=random_search.best_params_['C'],
            coef0=random_search.best_params_['coef0'],
            kernel=random_search.best_params_['kernel'],
            # gamma=random_search.best_params_['gamma'],
            degree=random_search.best_params_['degree'],
            epsilon=random_search.best_params_['epsilon'],
            shrinking=random_search.best_params_['shrinking'],
            tol=random_search.best_params_['tol'])
# endregion Tuning hyperparameters


# Fitting model with train data, then predicting on test data
svr_model_fit=svr_model.fit(X_train_scaled, y_train)
svr_pred = svr_model_fit.predict(X_test_scaled)

# Calculating the RMSE and R2 of the final model
svr_rmse=np.sqrt(mean_squared_error(y_test,svr_pred))
print('\nSVR RMSE:\n', svr_rmse, '\n')
svr_r2=r2_score(y_test,svr_pred)
print('SVR R2:\n', svr_r2, '\n')

# Plotting scatterplot of ground truth data vs predictions made by final (feature selected and hyperparameter tuned) model
plot_scatterplot(y_test,svr_pred,'Final SVR Model',ax_scp[1])
scatterplot_fig.suptitle('SVR Comparison of Ground Truth and Predicted Values', fontsize=18, fontweight='bold',y=0.96)
scatterplot_fig.savefig('SVR_Scatterplots.png',dpi=600)


# Plot the final learning curve with both feature selection and hyperparameter tuning 
plot_learning_curve(svr_model, X_train_scaled, y_train, np.linspace(0.1, 1.0, 10), 10, "After Feature Selection and Hyperparameter Tuning", ax[2])
learning_curves_fig.suptitle('Learning Curves for SVR', fontsize=18, fontweight='bold', y = 0.96)
learning_curves_fig.tight_layout()
learning_curves_fig.savefig("SVR_LearningCurves.png",dpi=600)




# region Evaluation scores to Regressor_eval_scores.csv

# specify the file name and new row data
filename = "Regressor_eval_scores.csv"
classifier = 'SVR'
new_row = [classifier, default_svr_r2, svr_r2, default_svr_rmse, svr_rmse]

# open the file in append mode
with open(filename, mode="a", newline="") as csvfile:

    # create a writer object
    writer = csv.writer(csvfile)

    # write the new row to the file
    writer.writerow(new_row)
# endregion Evaluation scores to Regressor_eval_scores.csv

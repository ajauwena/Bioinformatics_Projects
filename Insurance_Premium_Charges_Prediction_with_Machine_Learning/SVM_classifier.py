"""
===================
Regression with SVM (SVR)
===================
Author: Nishita Sharif, nishita.sharif@gmail.com
Date: April 21, 2023

How to run:   python3  SVM_classifier.py insurance_dataset_clean.csv

Output figures:
SVC_LearningCurves.png
SVC_Feature_Importances.png
SVC_Confusion_Matrices.png 
=================
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import randint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
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

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring= 'accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    axis.set_title(title)
    axis.set_xlabel("Training examples")
    axis.set_ylabel("Accuracy")
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



# ----------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------

# defining command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='The path to the file to process')
args = parser.parse_args()

clean_dataset      = args.input_file
insurance_premium_df = pd.read_csv(clean_dataset, header=0, encoding="utf8")


# Make a new column that contains the information from the smoker variable and charge classes to stratify from
insurance_premium_df["stratify_column"] = insurance_premium_df["smoker"].astype(str) + insurance_premium_df["charge_classes"].astype(str)

# Creating an object with the values in the data frame
data = insurance_premium_df.values

# Assign the data in all columns except the columns "charges", "charge_classes," and "stratify_column" as input variables.
X = data[:, :-3]

# Assign the data in the column "charge_classes" as the output variable.
y = data[:, -2].astype('int')

# Splitting the data into training and testing sets, then scaling (using the split_and_scale_data function created)
X_train_scaled,X_test_scaled,y_train, y_test=split_and_scale_data(X,y)
default_svc_pred= SVC().fit(X_train_scaled, y_train).predict(X_test_scaled)
default_svc_accuracy=accuracy_score(y_test, default_svc_pred)
default_svc_f1=f1_score(y_test, default_svc_pred, average='weighted')

# Plotting confusion matrix for default model
confusion_matrices, ax_cm = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
ax_cm[0].set_title('Default Model')
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, default_svc_pred)).plot(ax=ax_cm[0])
ax_cm[0].xaxis.set_ticklabels(['Low', 'Medium', 'High']); ax_cm[0].yaxis.set_ticklabels(['Low', 'Medium', 'High'])

# Creating the learning curves figure
learning_curves_fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
learning_curves_fig.subplots_adjust(hspace=0.3, wspace=0.13)

# Plot the final learning curve with the default (but scaled) data
plot_learning_curve(SVC(), X_train_scaled, y_train, np.linspace(0.1, 1.0, 10), 10, "Default", ax[0])



# region Feature Selection
svr = SVC().fit(X_train_scaled, y_train)
importance = permutation_importance(svr, X_train_scaled, y_train, n_repeats=10, random_state=23).importances_mean
feature_names = insurance_premium_df.columns[:-1][:-2]

# Use the zip() function to create a dictionary with feature names and their respective importance values
feats = {}
for feature, importance in zip(feature_names, importance):
    feats[feature] = importance #add the name/value pair 

# Create a dataframe using the tuple created above, then sort in descending order
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'}).sort_values(by='Importance', ascending=False)

# View feature importance
print("\nFeature importance for SVC:")
print(importances)

# plot feature importance
feature_importance_fig, ax_fs = plt.subplots()
ax_fs.bar(list(importances.index), importances.loc[:,'Importance'], color ="#9dc6d8")
ax_fs.set_xlabel('Features')
ax_fs.set_xticks(range(len(importances.index)))
ax_fs.set_xticklabels(list(importances.index), rotation=45)
ax_fs.set_ylabel('Importance')
ax_fs.set_title("SVC Feature Importance")
feature_importance_fig.tight_layout()
feature_importance_fig.savefig("SVC_Feature_Importances.png", dpi=600)

top_5_features = importances.index[:5]
X_selected_features = insurance_premium_df[top_5_features].values

# Splitting the data into training and testing sets, then scaling (using the split_and_scale_data function created)
X_train_scaled,X_test_scaled,y_train, y_test=split_and_scale_data(X_selected_features,y)

# Plot the final learning curve after feature selection
plot_learning_curve(SVC(), X_train_scaled, y_train, np.linspace(0.1, 1.0, 10), 10, "After Feature Selection", ax[1])

# endregion Feature Selection


# region tuning hyperparameters

# define the hyperparameter space to search over
param_space = {'C': randint(1, 100), 
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
              'degree': randint(1, 10),
              'gamma': ['scale', 'auto'],
              'shrinking': [True, False],
              'coef0': [0, 1, 2],
              'decision_function_shape': ['ovo', 'ovr'],
              'tol': [0.001, 0.0001]}


# define the randomized search
random_search = RandomizedSearchCV(SVC(), 
                                   param_distributions=param_space, 
                                   n_iter=100, 
                                   cv=5, 
                                   n_jobs=-1, 
                                   random_state=23)

# fit the randomized search to the training data
random_search.fit(X_train_scaled, y_train)


print("\nBest hyperparameters for SVC:")
for key, value in random_search.best_params_.items():
    print(f'{key}: {value}')
    
print(f"\nBest mean cross-validation score: {random_search.best_score_}")

svc_model = SVC(C=random_search.best_params_['C'],
            coef0=random_search.best_params_['coef0'],
            kernel=random_search.best_params_['kernel'],
            gamma=random_search.best_params_['gamma'],
            degree=random_search.best_params_['degree'],
            decision_function_shape=random_search.best_params_['decision_function_shape'],
            shrinking=random_search.best_params_['shrinking'],
            tol=random_search.best_params_['tol'])

svc_model_fit=svc_model.fit(X_train_scaled, y_train)

# endregion tuning hyperparameters



# Obtaining the accuracy and F1 scores
svc_pred = svc_model_fit.predict(X_test_scaled)
svc_accuracy = accuracy_score(y_test, svc_pred)
svc_precision = precision_score(y_test, svc_pred, average='weighted')
svc_f1 = f1_score(y_test, svc_pred, average='weighted')
svc_recall = recall_score(y_test, svc_pred, average='micro')
print('Accuracy: ', "%.4f" % (svc_accuracy))
print('Precision: ', "%.4f" % (svc_precision))
print('Recall: ', "%.4f" % (svc_recall))
print('F1: ', "%.4f" % (svc_f1))

# Plotting confusion matrix for final model
ax_cm[1].set_title('Final Model')
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, svc_pred)).plot(ax=ax_cm[1])
ax_cm[1].xaxis.set_ticklabels(['Low', 'Medium', 'High']); ax_cm[1].yaxis.set_ticklabels(['Low', 'Medium', 'High'])
confusion_matrices.suptitle('Confusion Matrices for SVC', fontsize=18, fontweight='bold', y = 0.96)
confusion_matrices.savefig("SVC_Confusion_Matrices.png", dpi = 600)

# Plot the final learning curve with both feature selection and hyperparameter tuning 
plot_learning_curve(svc_model, X_train_scaled, y_train, np.linspace(0.1, 1.0, 10), 10, "After Feature Selection and Hyperparameter Tuning", ax[2])

learning_curves_fig.suptitle('Learning Curves for SVC', fontsize=18, fontweight='bold', y = 0.96)
learning_curves_fig.tight_layout()
learning_curves_fig.savefig("SVC_LearningCurves.png", dpi = 600)


# region Evaluation scores to Classifier_eval_scores.csv

# specify the file name and new row data
filename = "Classifier_eval_scores.csv"
classifier = 'SVC'
new_row = [classifier, default_svc_accuracy, svc_accuracy, default_svc_f1,svc_f1]

# open the file in append mode
with open(filename, mode="a", newline="") as csvfile:

    # create a writer object
    writer = csv.writer(csvfile)

    # write the new row to the file
    writer.writerow(new_row)
# endregion Evaluation scores to Classifier_eval_scores.csv

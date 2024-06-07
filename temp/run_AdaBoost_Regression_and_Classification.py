"""
===================
Regression and classification with AdaBoost
===================
Author: Olusegun Odumosu, odumosu.segun@gmail.com
Date: April 21, 2023

How to run:   python3  run_AdaBoost_Regression_and_Classification.py -in insurance_dataset_clean.csv -k 10  -n 3

Output figures:
AdaBoost_Classification_LearningCurves.png
AdaBoost_Classification_Confusion_Matrices.png
AdaBoost_Classification_Performance_Summary.png
AdaBoost_Regression_Learning_Curves.png
AdaBoost_Regression_Scatterplots_Predictions.png
AdaBoost_Regression_Performance_Summary.png
=================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold, cross_val_score, cross_val_predict, learning_curve, RandomizedSearchCV
from sklearn import model_selection
from skopt import BayesSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score
from numpy import mean
import argparse
import sys
import joblib
import csv


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


# Return regressor/classifier and method name (for printing purposes)
method_name = {
    "AdaBoostR": "AdaBoost Regression",
    "AdaBoostC": "AdaBoost Classification",
}



# make scoring dictionary for classification
scoring_class = { 
    'Accuracy':'accuracy',
    'Balanced accuracy':'balanced_accuracy',
    'Precision':'precision_macro',
    'Recall':'recall_macro',
    'F1-score':'f1_macro'
}


# make scoring dictionary for regression
scoring_reg = { 
    'MAE':'neg_mean_absolute_error',
    'MSE':'neg_mean_squared_error',
    'RMSE':'neg_root_mean_squared_error',
    'R2':'r2',
    'Explained variance':'explained_variance',
    'Max Error':'max_error' 
}



# read data for regressor or classifier
def read_data():
    #load the dataset, header is first row
    insurance_premium_df = pd.read_csv(filename, header=0, encoding="utf8")
    # Make a new column that contains the information from the smoker variable and charge classes to stratify from
    insurance_premium_df["stratify_column"] = insurance_premium_df["smoker"].astype(str) + insurance_premium_df["charge_classes"].astype(str)
    data = insurance_premium_df.values
    # ---------------------for regression -----------------#
    # Assign the data in all columns except the columns "charges", "charge_classes" and "stratify_column" as input variables.
    X = data[:, :-3]
    #Assign the data in the column "charges" as the output variable.
    y_reg = data[:, -3]
    # Split the data using a 70:30 train:test split with stratified sampling.
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, train_size=0.70, random_state=24, stratify=insurance_premium_df['stratify_column']) 
    #scale the data
    scaler = StandardScaler()
    scaler.fit(X_train_reg)
    X_train_reg = scaler.transform(X_train_reg)
    X_test_reg = scaler.transform(X_test_reg)
    #-------------- for classification -----------------------#
    #Assign the data in the column "charge_classes" as the output variable.
    y_class = data[:, -2].astype("int")
    # Split the data using a 70:30 train:test split with stratified sampling.
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, train_size=0.70, random_state=24, stratify=insurance_premium_df['stratify_column'])
    #scale the data
    scaler = StandardScaler()
    scaler.fit(X_train_class)
    X_train_class = scaler.transform(X_train_class)
    X_test_class = scaler.transform(X_test_class)
    return insurance_premium_df, X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_train_class, X_test_class, y_train_class, y_test_class


# Set regressor/classifier model
# Return regressor/classifier and method name (for printing purposes)
def set_regressor_or_classifier(method):
    if (method == "AdaBoostR"):
        estimator = AdaBoostRegressor(random_state=24)

    elif (method == "AdaBoostC"):
        estimator = AdaBoostClassifier(random_state=24)

    else:
        print("\nError: Invalid method name:" + method + "\n")
        parser.print_help()
        sys.exit(0)
    return estimator, method_name[method]


# Evaluate model for regressor or classifier. Difference in scoring dict
def eval_model(method, estimator, num_sp, num_rep, X, y, step):
    kfold = RepeatedKFold(n_splits=num_sp, n_repeats=num_rep, random_state=24)
    num_characters = 20
    print("Model".ljust(num_characters),":", method_name[method])
    print("K-folds".ljust(num_characters),":", kf)
    print("Num splits".ljust(num_characters),":", num_splits)
    # for regressor
    if method == "AdaBoostR":
        for name,score in scoring_reg.items():
            results = model_selection.cross_val_score(estimator, X, y, cv=kfold, scoring=score, n_jobs=-1)
            print(name.ljust(num_characters), ": %.3f (%.3f)" % (np.absolute(results.mean()), np.absolute(results.std())))
            # add score to dictionary 
            if score == "r2":
                r2_reg[step] = results
            elif score == 'neg_root_mean_squared_error':
                RMSE_reg[step] = results
    # for classifier
    else:
        for name,score in scoring_class.items():
            results = model_selection.cross_val_score(estimator, X, y, cv=kfold, scoring=score, n_jobs=-1)
            print(name.ljust(num_characters), ": %.3f (%.3f)" % (np.absolute(results.mean()), np.absolute(results.std())))
            if score == 'accuracy':
                accuracy_class[step] = results
            elif score == 'f1_macro':
                f1_class[step] = results


# Plot learning curve for regressor or classifier. Difference in scoring and y label
def plot_learning_curve(method, estimator, title, X, y, cv, axis):
    # for regressor
    if method == "AdaBoostR":
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring= "r2")
    # for classifier
    else:
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring= "accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    axis.set_title(title)
    axis.set_xlabel("Training examples")
    # for regressor
    if method == "AdaBoostR":
        axis.set_ylabel("R$^2$ Score")
    # for classifier
    else:
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
   



# hyperparameter tune with bayes search for regressor. Difference in stratifiedKFold and params
def get_bayes_search(method, estimator, X, y):
    if method == "AdaBoostR":
        # define the param values to search
        params = {
            'n_estimators': list(range(1,501,25)),
            'learning_rate': list(np.arange(0.001,2,0.01)),
            'loss': ["linear", "square", "exponential"]
        }
        # define the evaluation procedure
        cv = RepeatedKFold(n_splits=kf, n_repeats = num_splits, random_state=24)
    else:
        params = {
        'n_estimators': list(range(1,501,25)),
        'learning_rate': list(np.arange(0.001,2,0.01)),
        'algorithm': ["SAMME", "SAMME.R"]
        }
        cv = RepeatedStratifiedKFold(n_splits=kf, n_repeats = num_splits, random_state=24)
    # define the bayes search procedure
    bayes_search = BayesSearchCV(estimator, search_spaces=params, n_jobs=-1, cv=cv, random_state=24)
    # execute the bayes search
    bayes_result = bayes_search.fit(X, y)
    # summarize the best score and configuration
    print("Best hyperparameters: %f using %s" % (bayes_result.best_score_, bayes_result.best_params_))
    return bayes_result

# get feature importance with permuatation feature importance
def get_and_plot_permutation_feature_importance(estimator, X, y, title, ax_fs):
    estimator.fit(X, y)
    results = permutation_importance(estimator, X, y, n_repeats=num_splits, random_state=24)
    # get importance
    importance = results.importances_mean
    # get feature names
    feature_names = insurance_premium_df.columns[0:8]
    # create a dictionary with feature names and importance values
    feature_importance = {}
    for feature, importance in zip(feature_names, importance):
        feature_importance[feature] = importance
    # create data frame of dictionary, reorient dictionary axis and sort by ascending order
    df_feature_importance = pd.DataFrame.from_dict(feature_importance, orient='index').rename(columns={0: 'Importance'}).sort_values(by='Importance', ascending=False)
    print("\n\nPermutation feature importance scores")
    print(df_feature_importance)
    # select features with importance above 0.00001
    selected_features = dict((k,v) for k,v in feature_importance.items() if v >=0.00001) 
    # create a list of indices of selected features
    selected_features_indices = [insurance_premium_df.columns.get_loc(key) for key in selected_features]    
    #plot feature importance
    ax_fs.bar(list(df_feature_importance.index), df_feature_importance.loc[:,'Importance'], color ="#9dc6d8")
    ax_fs.set_xlabel('Features')
    ax_fs.set_ylabel('Importance')
    ax_fs.set_title(title)
    return selected_features, selected_features_indices
    

def concordance_correlation_coefficient(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    s_xy = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean)) / (len(y_true) - 1)
    s_x2 = np.sum((y_true - y_true_mean) ** 2) / (len(y_true) - 1)
    s_y2 = np.sum((y_pred - y_pred_mean) ** 2) / (len(y_true) - 1)
    r = s_xy / np.sqrt(s_x2 * s_y2)
    # mse = mean_squared_error(y_true, y_pred)
    ccc = 2 * r * np.std(y_true) * np.std(y_pred) / (np.var(y_true) + np.var(y_pred) + (y_true_mean - y_pred_mean) ** 2)
    return ccc

# Regression prediction plot - Plot predicted values against true values
def plot_predictions_reg(regressor, num_sp, title, X, y, ax_pr):
    predicted = cross_val_predict(regressor, X, y, cv=num_sp, n_jobs=-1)
    # fig, ax = plt.subplots()
    ax_pr.scatter(y, predicted, s=15, color = "#7dd0b6")
    # plot 45 degree line for perfect predictions
    ax_pr.plot([y.min(), y.max()], [y.min(), y.max()], 'k-')

    m, b = np.polyfit(y_test_class.astype(float), y_test_class.astype(float), 1)
    # plot line of best fit for actual predictions
    m, b = np.polyfit(y.astype(float), predicted.astype(float), 1)
    ax_pr.plot(y, m*y + b, color='red',  label='Best fit')
    ax_pr.text(600,42000,f'Line of best fit: {m:.2f}x + {b:.2f}')
    ax_pr.set_xlabel('Ground Truth Values')
    ax_pr.set_ylabel('Predicted Values')
    ax_pr.set_title(title)
    # add R2 to plot
    r2 = r2_score(y, predicted)
    ccc = concordance_correlation_coefficient(y, predicted)
    ax_pr.legend()
    ax_pr.text(600,38000,f'R$^2$ = {r2:.4f}')
    ax_pr.text(600,40000,f'CCC = {ccc:.4f}')



# classifier prediction plot
def plot_confusion_matrix(classifier, num_sp, title, X, y, axis_number):
    predicted = cross_val_predict(classifier, X, y, cv=num_sp, n_jobs=-1)
    ax_cm[axis_number].set_title(title)
    ConfusionMatrixDisplay( confusion_matrix=confusion_matrix(y, predicted)).plot(ax=ax_cm[axis_number])
    ax_cm[axis_number].xaxis.set_ticklabels(['Low', 'Medium', 'High']); 
    ax_cm[axis_number].yaxis.set_ticklabels(['Low', 'Medium', 'High'])
    

# plot summary of evlautions at different stages
def plot_summary(dict, score, title, ax_sc):
    # extract dictionary keys as list
    step = list(dict.keys())
    result = list(dict.values())
    ax_sc.boxplot(result, labels = step, showmeans = True)
    ax_sc.set_xticklabels(step, rotation=45, ha='right')
    ax_sc.set_xlabel('Step')
    if (score == "r2"):
        ax_sc.set_ylabel("R$^2$ Score")
    elif (score == "F1_score"):
        ax_sc.set_ylabel("F1 Score")
    else:
        ax_sc.set_ylabel(score)
    ax_sc.set_title(title)








# -------------------------------------------------------------------------------------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------------------------------------------------------------------------------------
# define command line arguments
parser = argparse.ArgumentParser(description='AdaBoost regression and classification')
parser.add_argument('--in_file', '-in', action="store", dest ='in_file', default='insurance_dataset_clean.csv', required=False, help='Name of csv input file. The last colum of the file is the desired output.')
parser.add_argument('--kfold', '-k', action="store", dest ='kfold', default=10, required=False, help='Number of folds for cross-validation')
parser.add_argument('--num_splits', '-n', action="store", dest='num_splits', default=3, required=False, help='Number of randomly permutted splits for cross-validation')


# handle user errors
try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
    
    
filename      = args.in_file
kf            = int(args.kfold)
num_splits    = int(args.num_splits)
method_reg    = "AdaBoostR"
method_class  = "AdaBoostC"

# initialize dictionary for scores
r2_reg ={}
RMSE_reg = {}
f1_class = {}
accuracy_class ={}

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Analysis for regression
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\n\n-----------------------------ADABOOST REGRESSION ANALYSIS--------------------------------------")
# set regressor
regressor, method_name_reg = set_regressor_or_classifier(method_reg)

# load data from file
insurance_premium_df, X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_train_class, X_test_class, y_train_class, y_test_class = read_data()

# evaluate model
regressor.fit(X_train_reg, y_train_reg)
eval_model(method_reg, regressor, kf, num_splits, X_train_reg, y_train_reg, step ="Default")

# initialize grid plot for learning curves
learning_curves_fig_reg, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
learning_curves_fig_reg.subplots_adjust(hspace=0.3, wspace=0.7)

# plot learning curve
title1 = r"Default Model"
plot_learning_curve(method_reg, regressor, title1, X_train_reg, y_train_reg, kf, ax[0,0])

# hyperparameter tune with bayes search
print("\n\n Bayes search hyperparameter best results regression")
bayes_result_reg = get_bayes_search(method_reg, regressor, X_train_reg, y_train_reg)
#set paramters for updated regressor
regressor_updated = regressor.set_params(**bayes_result_reg.best_params_)
regressor_updated.fit(X_train_reg, y_train_reg)

# evaluate updated regressor
print("\n\nModel Evaluation after hyperparameter tuning")
eval_model(method_reg, regressor_updated, kf, num_splits, X_train_reg, y_train_reg, step = "Hyperparameter_tuned")


#plot learning curve for updated regressor
title2 = r"After Hyperparameter Tuning" + method_name_reg
plot_learning_curve(method_reg, regressor_updated, title2, X_train_reg, y_train_reg, kf, ax[0,1])


## Feature selection
# initialize grid plot for feature selection
feature_selection_fig, ax_fs = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
feature_selection_fig.subplots_adjust(hspace=0.3, wspace=0.13)
# check importance of features with permutation feature importance
title3 = method_name_reg
selected_features_reg, selected_features_reg_indices = get_and_plot_permutation_feature_importance(regressor_updated, X_train_reg, y_train_reg, title3, ax_fs[0])
print("Selected features  importance scores regressor: ", selected_features_reg)

# select important features
X_train_reg_selected = np.take(X_train_reg, selected_features_reg_indices, axis=1)
X_test_reg_selected = np.take(X_test_reg, selected_features_reg_indices, axis=1)
# # print("Dimensions of original X ", X_train_reg.shape)
# # print("Dimensions of updated X :", X_train_reg_selected.shape)

# Evaluate model after feature selection
print("\n\nModel Evaluation after Feature selection")
eval_model(method_reg, regressor_updated, kf, num_splits, X_train_reg_selected, y_train_reg, step = "feature_selected")   

# learning curve after feature selection
title4 = r"After Feature Selection"
plot_learning_curve(method_reg, regressor_updated, title4, X_train_reg_selected, y_train_reg, kf, ax[1,0])

# hyperparameter tune with bayes search after feature selection
print("\n\n Bayes search hyperparameter best results after feature selection regression")
bayes_result_reg2 = get_bayes_search(method_reg, regressor_updated, X_train_reg_selected, y_train_reg)
#set paramters for updated regressor
regressor_updated2 = regressor.set_params(**bayes_result_reg2.best_params_)
regressor_updated2.fit(X_train_reg_selected, y_train_reg)

# evaluate updated regressor
print("\n\nModel Evaluation after feature selection and hyperparameter tuning")
eval_model(method_reg, regressor_updated2, kf, num_splits, X_train_reg_selected, y_train_reg, step = "feature_selected_and_tuned")


# plot learning curve for updated regressor
title5 = r"After Feature Selection and Hyperparameter Tuning"
plot_learning_curve(method_reg, regressor_updated2, title5, X_train_reg_selected, y_train_reg, kf, ax[1,1])


learning_curves_fig_reg.suptitle('Learning Curves for AdaBoost Regression', fontsize=18, fontweight='bold', y = 0.97)
learning_curves_fig_reg.tight_layout()
learning_curves_fig_reg.savefig("AdaBoost_Regression_Learning_Curves.png",dpi=600)
learning_curves_fig_reg.show


############ Testing  ####################
# evaluate model with test dataset all features
print("\n\nModel Evaluation test dataset all features")
eval_model(method_reg, regressor_updated, kf, num_splits, X_test_reg, y_test_reg, step = "test_all_features")

# evaluate model with test dataset selected features
print("\n\nModel Evaluation test dataset selected features")
eval_model(method_reg, regressor_updated2, kf, num_splits, X_test_reg_selected, y_test_reg, step = "test_selected_features")

#plot predictions with new model for selected features
prediction_reg_fig, ax_pr = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
prediction_reg_fig.subplots_adjust(hspace=0.3, wspace=0.7)
title6 = "With Selected Features"
plot_predictions_reg(regressor_updated2, num_splits, title6, X_test_reg_selected, y_test_reg, ax_pr[1])
    
# plot predictions with older tuned model for all features
title7 = "With All Features"
plot_predictions_reg(regressor_updated, num_splits, title7, X_test_reg, y_test_reg, ax_pr[0])

prediction_reg_fig.suptitle('Scatterplot of Charge Predictions with AdaBoost Regressor', fontsize=18, fontweight='bold', y = 0.97)
prediction_reg_fig.tight_layout()
prediction_reg_fig.savefig("AdaBoost_Regression_Scatterplots_Predictions.png",dpi=600)
prediction_reg_fig.show()    
############## Plot summary of scores #############    
summary_reg_fig, ax_sc = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
summary_reg_fig.subplots_adjust(hspace=0.3, wspace=0.7)
title16 = r"R$^2$ Score" 
plot_summary(r2_reg, "r2", title16, ax_sc[0])
title17 = r"RMSE" 
plot_summary(RMSE_reg, "RMSE", title17, ax_sc[1])    
summary_reg_fig.suptitle('R$^2$ and RMSE of AdaBoost Regression at different steps', fontsize=18, fontweight='bold', y = 0.96)
summary_reg_fig.tight_layout()
summary_reg_fig.savefig("AdaBoost_Regression_Performance_Summary.png",dpi=600)
summary_reg_fig.show()

############### save models ################ 
## model with all features
regressor_updated.fit(X_train_reg, y_train_reg)
filename3 = 'AdaBoost_Regression_finalized_model_all_features.sav'
joblib.dump(regressor_updated, filename3)

## model with selected features
regressor_updated2.fit(X_train_reg_selected, y_train_reg)
filename4 = 'AdaBoost_Regression_finalized_features_age_BMI_smoke.sav'
joblib.dump(regressor_updated2, filename4)














# # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Classification analysis
# # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\n\n---------------------------------------------ADABOOST CLASSIFICATION ANALYSIS--------------------------------------------")
# set regressor
classifier, method_name_class = set_regressor_or_classifier(method_class )

# evaluate model
classifier.fit(X_train_class, y_train_class)
eval_model(method_class, classifier, kf, num_splits, X_train_class, y_train_class, step ="Default")

# initialize grid plot for learning curves
learning_curves_fig_class, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
learning_curves_fig_class.subplots_adjust(hspace=0.3, wspace=0.7)

# plot learning curve
title8 = r"Default Model"
plot_learning_curve(method_class, classifier, title8, X_train_class, y_train_class, kf, ax2[0,0])

# hyperparameter tune with bayes search
print("\n\n Bayes search hyperparameter best results classification")
bayes_result_class = get_bayes_search(method_class, classifier, X_train_class, y_train_class)
#set paramters for updated classifier
classifier_updated = classifier.set_params(**bayes_result_class.best_params_)
classifier_updated.fit(X_train_class, y_train_class)

# evaluate updated classifier
print("\n\n Model Evaluation after hyperparameter tuning")
eval_model(method_class, classifier_updated, kf, num_splits, X_train_class, y_train_class, step = "Hyperparameter_tuned")


#plot learning curve for updated classifier
title9 = r"After Hyperparameter Tuning"
plot_learning_curve(method_class, classifier_updated, title9, X_train_class, y_train_class, kf, ax2[0,1])


## Feature selection
# check importance of features with permutation feature importance
title10 = method_name_class
selected_features_class, selected_features_class_indices = get_and_plot_permutation_feature_importance(classifier_updated, X_train_class, y_train_class, title10, ax_fs[1])
feature_selection_fig.suptitle('Feature Importances for AdaBoost Regressor and Classifier', fontsize = 18, fontweight='bold', y = 0.96)
feature_selection_fig.tight_layout()
feature_selection_fig.savefig("AdaBoost_Feature_Importances.png",dpi=600)
print("Selected features importance scores classifier: ", selected_features_class)
# select important features
X_train_class_selected = np.take(X_train_class, selected_features_class_indices, axis=1)
X_test_class_selected = np.take(X_test_class, selected_features_class_indices, axis=1)


# Evaluate model after feature selection
print("\n\nModel Evaluation after Feature selection")
eval_model(method_class, classifier_updated, kf, num_splits, X_train_class_selected, y_train_class, step = "feature_selected")   

# learning curve after feature selection
title11 = r"After Feature Selection"
plot_learning_curve(method_class, classifier_updated, title11, X_train_class_selected, y_train_class, kf, ax2[1,0])

# hyperparameter tune with bayes search after feature selection
print("\n\n Bayes search hyperparameter best results after feature selection")
bayes_result_class2 = get_bayes_search(method_class, classifier_updated, X_train_class_selected, y_train_class)
#set paramters for updated classifier
classifier_updated2 = classifier.set_params(**bayes_result_class2.best_params_)
classifier_updated2.fit(X_train_class_selected, y_train_class)

# evaluate updated classifier
print("\n\nModel Evaluation after feature selection and hyperparameter tuning")
eval_model(method_class, classifier_updated2, kf, num_splits, X_train_class_selected, y_train_class, step = "feature_selected_and_tuned")


# plot learning curve for updated classifier
title12 = r"After Feature Selection and Hyperparameter Tuning "
plot_learning_curve(method_class, classifier_updated2, title12, X_train_class_selected, y_train_class, kf, ax2[1,1])

learning_curves_fig_class.suptitle('Learning Curves for AdaBoost Classification', fontsize=18, fontweight='bold', y = 0.96)
learning_curves_fig_class.tight_layout()
learning_curves_fig_class.savefig("AdaBoost_Classification_LearningCurves.png",dpi=600)
learning_curves_fig_class.show()


############ Testing  ####################
# evaluate model with test dataset all features
print("\n\nModel Evaluation test dataset all features")
eval_model(method_class, classifier_updated, kf, num_splits, X_test_class, y_test_class, step = "test_all_features")

# evaluate model with test dataset selected features
print("\n\nModel Evaluation test dataset selected features")
eval_model(method_class, classifier_updated2, kf, num_splits, X_test_class_selected, y_test_class, step = "test_selected_features")

## plot confusion_matrix ##
# initialize grid plot for learning curves
confusion_matrix_fig, ax_cm = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
confusion_matrix_fig.subplots_adjust(hspace=0.3, wspace=0.7)
# model with all features
title13 = r"With All Features"
plot_confusion_matrix(classifier_updated, num_splits, title13, X_test_class, y_test_class, axis_number=0)
# model with selected features
title14 = r"With Selected Features"
plot_confusion_matrix(classifier_updated2, num_splits, title14, X_test_class_selected, y_test_class, axis_number=1)

confusion_matrix_fig.suptitle('Confusion Matrices for AdaBoost Classification', fontsize=18, fontweight='bold', y = 0.96)
confusion_matrix_fig.tight_layout()
confusion_matrix_fig.savefig("AdaBoost_Classification_Confusion_Matrices.png",dpi=600)
confusion_matrix_fig.show()

#################### plot summary plots #################
summary_class_fig, ax_sc = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
summary_class_fig.subplots_adjust(hspace=0.3, wspace=0.7)
title19 = r"Accuracy"
plot_summary(accuracy_class, "Accuracy", title19, ax_sc[0])
title20 = r"F1 Scores" 
plot_summary(f1_class, "F1_score", title20, ax_sc[1])    
summary_class_fig.suptitle('Accuracy and F1 score Of AdaBoost Classifier at different steps', fontsize=18, fontweight='bold', y = 0.96)
summary_class_fig.tight_layout()
summary_class_fig.savefig("AdaBoost_Classification_Performance_Summary.png", dpi=600)
summary_class_fig.show()
#################### save models ###########################
## model with all features
classifier_updated.fit(X_train_class, y_train_class)
filename5 = 'AdaBoost_classifier_finalized_model_all_features.sav'
joblib.dump(classifier_updated, filename5)

## model with selected features
classifier_updated2.fit(X_train_class_selected, y_train_class)
filename6 = 'AdaBoost_classifier_finalized_selected_features.sav'
joblib.dump(classifier_updated2, filename6)


#--------------------------------------------------------------------------------------------------------------------------------------------
# append first and last score to csv file for comparison with other models 
#--------------------------------------------------------------------------------------------------------------------------------------------
##### For regressor #####
# specify the file name and new row data
filename = "Regressor_eval_scores.csv"
classifier = 'AdaBoost'
new_row = [classifier, mean(r2_reg["Default"]), mean(r2_reg["feature_selected_and_tuned"]), np.absolute(mean(RMSE_reg["Default"])), np.absolute(mean(RMSE_reg["feature_selected_and_tuned"]))]
# open the file in append mode
with open(filename, mode="a", newline="") as csvfile:

    # create a writer object
    writer = csv.writer(csvfile)

    # write the new row to the file
    writer.writerow(new_row)
    
##### For classfier #####
filename2 = "Classifier_eval_scores.csv"
classifier2 = 'AdaBoost'
new_row2 = [classifier2, mean(accuracy_class["Default"]) , mean(accuracy_class["feature_selected_and_tuned"]), mean(f1_class["Default"]), mean(f1_class["feature_selected_and_tuned"])]

# open the file in append mode
with open(filename2, mode="a", newline="") as csvfile:

    # create a writer object
    writer = csv.writer(csvfile)

    # write the new row to the file
    writer.writerow(new_row2)
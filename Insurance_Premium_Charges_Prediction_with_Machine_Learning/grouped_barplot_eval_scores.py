"""
=========================================
Comparing Models' Performance
=========================================
Authors:
    Nishita Sharif, nishita.sharif@gmail.com
Date: April 21, 2023

How to run:   python3  grouped_barplot_eval_scores.py Regressor_eval_scores.csv Classifier_eval_scores.csv

Output figures:
Regression_Model_Comparison.png
Classification_Model_Comparison.png
=========================================
"""
# Importing modules
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


# ----------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------

# Define comand line arguments
parser = argparse.ArgumentParser()
parser.add_argument('input_files', nargs=2, help='Paths to the input files')
args = parser.parse_args()

# Assigning the command line arguments to objects
regression_scores_csv      = args.input_files[0]
classification_scores_csv      = args.input_files[1]

# Reading in and obtaining values from the regressor .csv file
regressor_eval_scores_df = pd.read_csv(regression_scores_csv, header=0, encoding="utf8")
regressor_eval_scores = regressor_eval_scores_df.values

# Getting a list of the regressor models
regressor_list = regressor_eval_scores_df['Regressor'].tolist()

# Creating figure that plots the grouped barplots comparing regressor models
regressor_evals_fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
x = np.arange(2)
width = 0.2
ax[0].bar(x-0.2, regressor_eval_scores[0][1:3], width, color='#7dd0b6')
ax[0].bar(x, regressor_eval_scores[1][1:3], width, color='#e38690')
ax[0].bar(x+0.2, regressor_eval_scores[2][1:3], width, color='#00b3ca')
ax[0].set_xticks(x)
ax[0].set_xticklabels(['Original', 'Final'])
ax[0].set_ylabel("Scores")
ax[0].set_title('R$^2$', fontdict={'fontsize': 15})
ax[0].legend(regressor_list)

ax[1].bar(x-0.2, regressor_eval_scores[0][3:5], width, color='#7dd0b6')
ax[1].bar(x, regressor_eval_scores[1][3:5], width, color='#e38690')
ax[1].bar(x+0.2, regressor_eval_scores[2][3:5], width, color='#00b3ca')
ax[1].set_xticks(x)
ax[1].set_xticklabels(['Default', 'Final'])
ax[1].set_ylabel("Scores")
ax[1].set_title('RMSE', fontdict={'fontsize': 15})
ax[1].legend(regressor_list)
regressor_evals_fig.suptitle('Regression Model Comparison', fontsize=18, fontweight='bold', y = 0.96)
regressor_evals_fig.tight_layout()
regressor_evals_fig.savefig("Regression_Model_Comparison.png",dpi=600)




# Reading in and obtaining values from the classifier .csv file
classifier_eval_scores_df = pd.read_csv(classification_scores_csv, header=0, encoding="utf8")  
classifier_eval_scores = classifier_eval_scores_df.values

# Getting a list of the classifier models
classifier_list = classifier_eval_scores_df['Classifier'].tolist()

# Creating figure that plots the grouped barplots comparing regressor models
classifier_evals_fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
x = np.arange(2)
width = 0.2
axes[0].bar(x-0.2, classifier_eval_scores[0][1:3], width, color='#7dd0b6')
axes[0].bar(x, classifier_eval_scores[1][1:3], width, color='#e38690')
axes[0].bar(x+0.2, classifier_eval_scores[2][1:3], width, color='#00b3ca')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Default', 'Final'])
axes[0].set_ylabel("Scores")
axes[0].set_title('Accuracy', fontdict={'fontsize': 15})
axes[0].legend(classifier_list)

axes[1].bar(x-0.2, classifier_eval_scores[0][3:5], width, color='#7dd0b6')
axes[1].bar(x, classifier_eval_scores[1][3:5], width, color='#e38690')
axes[1].bar(x+0.2, classifier_eval_scores[2][3:5], width, color='#00b3ca')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Original', 'Final'])
axes[1].set_ylabel("Scores")
axes[1].set_title('F1', fontdict={'fontsize': 15})
axes[1].legend(classifier_list)
classifier_evals_fig.suptitle('Classification Model Comparison', fontsize=18, fontweight='bold', y = 0.96)
classifier_evals_fig.tight_layout()
classifier_evals_fig.savefig("Classification_Model_Comparison.png", dpi = 600)
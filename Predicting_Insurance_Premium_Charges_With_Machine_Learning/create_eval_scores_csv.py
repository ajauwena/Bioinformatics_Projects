"""
=========================================================
Creating .csv files to store Evaluation Scores of Models
=========================================================
Author: Nishita Sharif, nishita.sharif@gmail.com
Date: April 21, 2023

How to run:   python3  create_eval_scores_csv.py 

Output Files:
Classifier_eval_scores.csv
Regressor_eval_scores.csv
=======================================================
"""

import csv

# specify the file name and column headers
filename = "Classifier_eval_scores.csv"
headers = ["Classifier", "original_Accuracy", "Accuracy", "original_F1", "F1"]

# create and open the file in write mode
with open(filename, mode="w", newline="") as csvfile:

    # create a writer object
    writer = csv.writer(csvfile)

    # write the column headers to the file
    writer.writerow(headers)



filename = "Regressor_eval_scores.csv"
headers = ["Regressor", "original_R2","R2","original_RMSE","RMSE"]

# create and open the file in write mode
with open(filename, mode="w", newline="") as csvfile:

    # create a writer object
    writer = csv.writer(csvfile)

    # write the column headers to the file
    writer.writerow(headers)
    
# Classifiers: Accuracy, F1 
# Regressors: R2, RMSE
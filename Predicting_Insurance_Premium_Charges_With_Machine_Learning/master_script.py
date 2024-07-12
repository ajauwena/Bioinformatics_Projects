"""
==============================================================================================
Predicting Insurance Premium Charges: An Exploration and Comparison of Machine Learning Models
==============================================================================================
Authors:
    Abelhard Jauwena, ajauwena@gmail.com
    Olusegun Odumosu, odumosu.segun@gmail.com
    Nishita Sharif, nishita.sharif@gmail.com
Date: April 21, 2023

How to run:   python3  master_script.py

Output files:
Classifier_eval_scores.csv
insurance_dataset_clean.csv
Regressor_eval_scores.csv

Output figures:
Characteristics_of_Dataset.png
Outliers_KNN_method.png
SVC_Feature_Importances.png
SVC_LearningCurves.png
SVC_Confusion_Matrices.png
SVR_Feature_Importances.png
SVR_LearningCurves.png
SVC_Scatterplots.png
AdaBoost_Classification_LearningCurves.png
AdaBoost_Classification_Confusion_Matrices.png
AdaBoost_Classification_Performance_Summary.png
AdaBoost_Regression_Learning_Curves.png
AdaBoost_Regression_Scatterplots_Predictions.png
AdaBoost_Regression_Performance_Summary.png
KNN_Classifier_Hyperparameter_Tuning.png
KNN_Classifier_Feature_Importances.png
KNN_Classifier_Confusion_Matrices.png
==============================================================================================
"""

import os


################ ACTUAL ORDER
os.system("python3  explore_and_visualize_initial_data.py -in insurance_dataset.csv")

os.system("python3  preparing_data.py -in insurance_dataset.csv")

os.system("python3 create_eval_scores_csv.py")

os.system("python3  SVM_classifier.py insurance_dataset_clean.csv")
os.system("python3  SVM_regressor.py insurance_dataset_clean.csv")

os.system("python3  run_AdaBoost_Regression_and_Classification.py -in insurance_dataset_clean.csv -k 10  -n 3")

os.system("python3 knn_regressor_hyperparameter_tuning.py -in insurance_dataset_clean.csv -k 1 -mk 50")
os.system("python3 knn_regressor_evaluation.py -in insurance_dataset_clean.csv -k 6 -mk 50")
os.system("python3 knn_classifier_hyperparameter_tuning.py -in insurance_dataset_clean.csv -k 5 -mk 50")
os.system("python3 knn_classifier_evaluation.py -in insurance_dataset_clean.csv -k 5 -mk 50")

os.system("python3  grouped_barplot_eval_scores.py Regressor_eval_scores.csv Classifier_eval_scores.csv")

os.system("python3 organize_files.py")
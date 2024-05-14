# Predicting Insurance Premium Charges: An Exploration and Comparison of Machine Learning Models

## Authors

Jauwena, A., Odumosu, O., and Sharif, N.

## Introduction

This project explores and compares various machine learning models to predict insurance premium charges based on six predictor variables: sex, age, BMI, smoker status, region, and number of children. Specifically, it constructs and evaluates the performance of AdaBoost, KNN, and SVM regressors and classifiers. 

The master script ("master_script.py") performs the following tasks:
- Data Exploration and Visualization    :   Generating plots that visualize the distribution of data.
- Data Cleaning                         :   Cleaning the data (removing missing or incorrect values, outlier detection and removal, encoding data, and shuffling data).
- Data Splitting                        :   Splitting the data into training and test data sets.
- Data Scaling                          :   Scaling predictor variable data.
- Permutation Feature Importance        :   Identifying features that are most important to prevent redundancy, prevent overfitting, and improve accuracy.
- Hyperparameter Optimization           :   Identifying the optimal hyperparameters for each model to improve performance of all models.
- Generating Learning Curves            :   Creating learning curves for all models throughout the construction of models.
- Model Testing                         :   The models are tested on unseen test data to evaluate model performance.

## Prerequisites

To run this script, the following must be installed:
- Python
- Matplotlib
- Numpy
- Pandas
- Scikit-learn

## Usage

The master script, which constructs and evaluates AdaBoost, KNN, and SVM regressors and classifiers, is not provided in this repository. Instead, the following scripts are provided, as they represent my own work and not my groupmates':
- "KNN_Regressor_Hyperparameter_Tuning.py".
- "KNN_Regressor_Evaluation.py".
- "KNN_Classifier_Hyperparameter_Tuning.py".
- "KNN_Classifier_Evaluation.py".

To run these scripts in the order listed above, execute these commands:
- "python3 knn_regressor_hyperparameter_tuning.py -in insurance_dataset_clean.csv -k 1 -mk 50"
- "python3 knn_regressor_evaluation.py -in insurance_dataset_clean.csv -k 6 -mk 50"
- "python3 knn_classifier_hyperparameter_tuning.py -in insurance_dataset_clean.csv -k 5 -mk 50"
- "python3 knn_classifier_evaluation.py -in insurance_dataset_clean.csv -k 5 -mk 50"

## Dataset

The dataset used in this analysis can be obtained from: https://www.kaggle.com/datasets/simranjain17/insurance
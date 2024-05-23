# Insurance Premium Charges Prediction: An Exploration and Comparison of Machine Learning Models

## Authors

Jauwena, A., Odumosu, O., and Sharif, N.

## Overview

This project explores and compares various machine learning models to predict insurance premium charges based on six predictor variables: sex, age, BMI, smoker status, region, and number of children. Specifically, it constructs and evaluates the performance of the "AdaBoost," "k-nearest neighbors" ("KNN"), and "support vector machine" ("SVM") regressors and classifiers. 

The master script ("master_script.py") performs the following tasks:

### 1. Exploring and Visualizing Data

Generates plots to visualize the distribution of data.

### 2. Cleaning Data

Cleans the data by removing missing or incorrect values, detecting and removing outliers, and encoding and shuffling data.

### 3. Splitting Data

Splits the data into training and testing sets.

### 4. Scaling Data

Scales the data for the predictor variable.

### 5. Permutation Feature Importance

Identifies the most important features to prevent redundancy and overfitting and improve accuracy.

### 6. Optimizing Hyperparameters

Identifies the optimal hyperparameters for each model to improve its performance.

### 7. Generating Learning Curves

Creates learning curves for all models throughout their construction.

### 8. Testing Models

Tests all models on unseen testing sets to evaluate their performances.

## Prerequisites

To run this script, the following libraries must be installed:
- Python.
- Matplotlib.
- Numpy.
- Pandas.
- Scikit-learn.

## Usage

More scripts, including the master script, will be added.

To run the master script, execute "python3  master_script.py".

## Dataset

The dataset used in this project can be obtained [here](https://www.kaggle.com/datasets/simranjain17/insurance).
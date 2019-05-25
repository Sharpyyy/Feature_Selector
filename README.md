# Feature_Selector
Extensive Feature reduction and Feature Selection using multiple Techniques

Hello Everyone,
In this project we will be showing various kinds of feature reduction and feature selection techniques for preprocessing of data before feedig it to the machine learning model.

## Getting Started
### Prerequisites

1. Install and setup Anaconda
Find an easy installation and setup guide using this [link](https://www.datacamp.com/community/tutorials/installing-anaconda-windows)
Make sure you install Anaconda for python 3.6 or above

2. Install the required packages:
Open Anaconda prompt and run these commands
```
conda install pandas, numpy, matplotlib, seaborn, xgboost, catboost, scikit-learn, statsmodels
```
Verify Installation by running
```
import pandas as pd
pd.__version__
```

3. Download the data sets:
We will be using **Superconductivty Data Data Set**. The goal here is to predict the critical temperature based on the features extracted. Data can be downloaded through this [link](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data)
 
 4. Clone my repository and unzip the repository into a folder
 
### How to run?
- Open 'Anaconda Prompt' from Start Menu
- Change working directory to the 'Feature_Selector' folder
- run this command to open the jupyter notebook
  `jupter notebook`
- open the ***Feature_Section_Regression.ipynb*** file run the cells one by one


## Some Theory
## Techniques used for Feature reduction/Selection:

### 1. Univariate feature selection
- Univariate feature selection works by selecting the best features based on univariate statistical tests.
- GenericUnivariateSelect allows to perform univariate feature selection with a configurable strategy. This allows to select the best univariate selection strategy with hyper-parameter search estimator.
This function take as input a scoring function that returns univariate scores and p-values.
- modes: 
    ```
    ‘percentile’ - removes all but a user-specified highest scoring percentage of features
    ‘k_best’ - removes all but the 'k' highest scoring features
    ‘fpr’ -  false positive rate
    ‘fdr’ - false discovery rate
    ‘fwe’ - family wise error
    ```
- score_fun :
```
    For regression: f_regression, mutual_info_regression
    For classification: chi2, f_classif, mutual_info_classif
```
    
### 2. Backward Elimination using Statistical Significance
- This method used p-values for elemination of the feature.
- Significance level can be set using p_threshold.
- We have used OLS(Ordinary Least Squares) regression (commonly known as Linear Regression) for finding p-values

### 3. Model-based  (Select-from-Model)
- SelectFromModel is a meta-transformer that can be used along with any estimator that has a coef_ or feature_importances_ attribute after fitting.
- The features are considered unimportant and removed, if the corresponding coef_ or feature_importances_ values are below the provided threshold parameter. Apart from specifying the threshold numerically, there are built-in heuristics for finding a threshold using a string argument.
- Available heuristics are “mean”, “median” and float multiples of these like “0.1*mean”

### 4. RFE (Recursive feature elimination) and RFE-CV (Recursive feature elimination with Cross Validation)
- Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features.
- First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_immportances_ attribute. Then, the least important features are pruned from current set of features.
- That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
  
### 5. 'feature_selector' module
- Feature selector is a tool for dimensionality reduction of machine learning datasets.

Technique available to identify features to remove:
- Missing Values
- Single Unique Values
- Collinear Features
- Zero Importance Features
- Low Importance Features

## Analysis of Model score vs the number of features using the RFE

In this analysis we have used the values for feature importance gained from the RFE technique and the number of features selected by RFE-CV technique. We then plot the model score vs the number of features selection in the descending order of the importances.
The plots are shown below:

### 1. Linear Regression
![Linear Regression Graph]()
### 2. Lasso Regression
![Lasso Regression Graph]()
### 3. Decision Tree Regression
![Decision Tree Regression Graph]()
### 4. Extra Trees Regression
![ExtraTreesRegressor Graph]()
### 5. XGB Regressor
![XGB Regressor Graph]()

## Project Premise 
This project is currently in progress, and will be updated throughout its development. Changes to this readme will continue to be made so that the most up-to-date descriptions of the code and findings can be found here.

This project explores how to evaluate different risk profiles from a retail banking perspective to identify which clients are most at-risk of defaulting on home loans. It uses the [Home Credit Group ](https://storage.googleapis.com/341-home-credit-default/home-credit-default-risk.zip) dataset to model these risk profiles using a set of binary classifiers.

## Project Approach

There are multiple very large files in this dataset, so for concision only the EDA and modeling within the application_train csv file are included in the modeling notebook. Automated pipelines for imputation, scaling, and model fitting are employed to set a baseline performance
measure for multiple ensemble classifiers, before employing the SMOTE algorithm to strategically oversample and undersample the dataset and refit models. 

These models are assessed against ROC-AUC, Matthews correlation coefficient, precision, recall, fbeta scores, and balanced accuracy metrics to have a balanced view of model performance across classes, given  that target is highly imbalanced. 

## Project Findings
## Recommendations for Further Research 

## Relevant Files
Please check out the loan_functions.py file to see the helper functions used, and requirements.txt for module/package information. Relevant dataset can be downloaded [here](https://storage.googleapis.com/341-home-credit-default/home-credit-default-risk.zip)

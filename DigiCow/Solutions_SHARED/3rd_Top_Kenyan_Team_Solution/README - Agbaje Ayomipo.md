# DigiCow Adoption Prediction Challenge -- 9th Place Solution

## Overview

This project predicts the probability that a farmer will adopt a
practice within 7 days of their first training session. The objective is
to enable early identification of farmers likely to turn training into
action, allowing DigiCow to prioritise follow-ups and tailor extension
strategies effectively.

Final Position: **9th Place**\
Total Pipeline Runtime: \~1 hour 40 minutes

------------------------------------------------------------------------

## Libraries Used

The following libraries were used in this solution:

numpy==2.2.6\
pandas==2.3.0\
tqdm==4.67.1\
scikit-learn==1.7.0\
feature-engine==1.8.3\
xgboost==3.0.2\
catboost==1.2.8\
lightgbm==4.6.0\
scipy==1.16.0\
sentence-transformers==5.2.2

Only the required libraries were imported and initialized to reduce
complexity and runtime.

A `requirements.txt` file is included with exact versions for reproducibility.
please run the command `pip install -r requirements.txt` when in the directory

------------------------------------------------------------------------

## Coding Environment

The solution was executed in a Jupyter Notebook environment.

To reproduce results: 
1. Ensure Python 3.10+ is installed. 
2. Install dependencies from `requirements.txt`.
3. Run the notebook from top to bottom.
4. An active internet connection is required (for sentence-transformers model downloads).

------------------------------------------------------------------------

## Data Used

Only the datasets provided on the competition page were used.\
No external datasets were incorporated.

The target variable: 
- `adopted_within_07_days` 
- `adopted_within_90_days`
- `adopted_within_120_days`

------------------------------------------------------------------------

## Data Processing

All preprocessing steps are fully reproducible within the notebook.

Processing steps include: - Dropping duplicates using `feature-engine` -
Encoding categorical variables - Ensuring consistent feature alignment
between train and test

No preprocessing was performed outside the notebook.

------------------------------------------------------------------------

## Cross-Validation & Training Strategy

A 10-fold StratifiedKFold cross-validation approach was used.

Key Function: `fit_predict()`

This function: - Performs 10-fold stratified cross-validation - Trains
the estimator - Computes Log Loss and ROC-AUC per fold - Generates
out-of-fold validation metrics - Averages predictions across folds for
final test prediction

Core logic: - StratifiedKFold (n_splits=10, shuffle=True,
random_state=2026) - Evaluation metrics: Log Loss and ROC-AUC - Final
prediction: Mean of fold predictions
Final Model: VotingClassifier()

------------------------------------------------------------------------

## Accessibility

All libraries used are freely available and publicly accessible. No
proprietary tools or paid services were used.

------------------------------------------------------------------------

## Coding Practices

-   Functions are modular and reusable.
-   Clear variable naming conventions were followed.
-   Repeated logic was encapsulated inside functions.
-   Feature names are descriptive and meaningful.

------------------------------------------------------------------------

## Exploratory Data Analysis (EDA)

An additional notebook containing detailed EDA has been included in the
submission folder.

The EDA provides: - Target distribution analysis - Feature
distributions - Correlation insights - Preliminary modeling insights

This is not required for leaderboard ranking but provides valuable
business insight.

------------------------------------------------------------------------

## Reproducibility Instructions

To reproduce results:

1.  Install dependencies using: pip install -r requirements.txt

2.  Ensure internet connection is active.

3.  Run the notebook from top to bottom without skipping cells.

Total expected runtime: \~1 hour 40 minutes.

------------------------------------------------------------------------

## Conclusion

This solution demonstrates a reproducible, modular, and production-ready
pipeline for predicting early farmer adoption behavior.

The approach ensures: - Strong validation strategy - Clean
preprocessing - Reproducibility - Practical deployment potential

Final Leaderboard Position: 9th Place
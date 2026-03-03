 Maize Price Forecasting Solution - Zindi

 1. Overview & Objectives
This solution provides a robust forecasting framework for maize prices in five key Kenyan counties (Kiambu, Kirinyaga, Mombasa, Nairobi, Uasin-Gishu). The objective is to predict market prices for Weeks 52 (2025) through Week 02 (2026).

Key Strategy:The solution utilizes a Cascading Prediction Architecture. Instead of treating counties in isolation, we model the market flow:
1.  Source Markets (e.g., Uasin-Gishu): Predicted first using only temporal features.
2.  Hub/Spoke Markets (e.g., Nairobi, Mombasa): Predicted subsequently, using the *predicted* prices of source markets as dynamic input features.

 2. Solution Architecture
The pipeline follows a strict modular design:
1.  ETL & Preprocessing: Aggregation of raw KAMIS and AgriBORA data.
2.  Bias Optimization Loop: A validation step on Weeks 46-51 to calibrate model bias before final forecasting.
3.  Recursive Inference: Step-by-step prediction where Week t predictions become features for Week t+1.

 Architecture Diagram
Data Flow: Raw CSVs -> Feature Engineering -> Train (Pre-W46) -> Optimize Bias (W46-W51) -> Final Forecast (W52-W02).

 3. ETL Process (Extract, Transform, Load)
* Data Sources:
      * `kamis_maize_prices_raw(1).csv`
      * `agriBORA_maize_prices(2).csv`
      * `agriBORA_maize_prices_weeks_46_to_51.csv`
[NOTE]*  Omitted kamis_maize_prices {too much noise and intruduced alot of undesired issues}
* Transformation Logic:
    * Grain Classification: Text mining applied to classification columns to standardize types into 'yellow', 'white', or 'traditional'.
    * Weighted Aggregation: Sources are weighted by reliability. AgriBORA data is given a weight of 100.0, AMIS sources 5.0, and others 1.0 during the weighted mean aggregation.
    * Imputation: Missing 'yellow' and 'traditional' prices are imputed using fixed ratios (0.8x and 1.1x of white maize price, respectively) to preserve row density.

 4. Data Modeling
* Algorithm: Gradient Boosting Regressor (`sklearn`).
* Hyperparameters:
    * `learning_rate`: 0.1
    * `n_estimators`: 500
    * `max_depth`: 3
    * `subsample`: 0.85
    * `random_state`: 42 (Ensures Reproducibility).
* Feature Engineering:
    * Temporal: `sin_mon`, `cos_mon`, `is_harvest_forced` (Nov/Dec flag).
    * Lag Features: Lags 1-4, Rolling Mean (4, 12 weeks), Momentum (Lag1 - Lag3).
    * Neighbor Features: Dynamic price lags from neighbor counties defined in `NEIGHBOR_MAP` (e.g., Nairobi uses Uasin-Gishu prices).

 5. Validation & Bias Optimization
A unique Bias Optimization step is implemented to handle recent market shifts:
1.  The model is trained on data prior to Week 46.
2.  It predicts on a validation set (Weeks 46-51).
3.  A grid search tests bias values from -2.0 to +2.0. The bias term that minimizes RMSE on the validation set is saved and applied to the final test forecast.

 6. Inference & Run Time
* Run Time: ~2 minutes on standard CPU.
* Inference Strategy: Recursive forecasting. The model predicts one week, updates the historical dataset with the prediction, and uses that updated history to predict the next week.
* Safety Floor: A minimum price floor of 15.0 is applied to prevent unrealistic drops.

 7. INSTRUCTIONS TO RUN
1.  Ensure all data files listed in Section 3 are in the root directory.
2.  Install dependencies: `pip install -r requirements.txt`.
3.  Run the solution: `python main.py` (or `python solution.py`).
4.  The script will output `submission.csv` containing the forecasts for Weeks 52, 01, and 02.


graph TD
GRAPH TD
    A[Raw Data Input] --> B{ETL Process}
    B -->|Clean & Weight| C[Feature Engineering]
    C -->|Generate Lags & Neighbors| D[Master Dataset]
    D --> E[Bias Optimization Loop]
    E -->|Train Pre-W46| F[Validate W46-W51]
    F -->|Calculate Best Bias| G[Final Model Config]
    G --> H[Recursive Forecasting W52-02]
    H -->|Step 1: Predict W52| I[Update History]
    I -->|Step 2: Predict W01| J[Update History]
    J -->|Step 3: Predict W02| K[Final Submission.csv]

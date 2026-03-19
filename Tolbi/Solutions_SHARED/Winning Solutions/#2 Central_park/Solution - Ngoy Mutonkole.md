# Team central_park solution

## Data preparation and feature engineering
Starting with the provided `TrainDataset.csv` file, we computed aggregated statistics (mean and standard deviation) for all bands per month.

The most important detail in this, is that we computed these stats based on a patch around the centre of the spectral images. We found that this patch based approach stabilized training and reduce the gap between CV and public LB scores from 0.4 to 0.02.

After experimenting with patch sizes, a value of patch size = 8 worked yielded the best CV scores and was thus used throughout modelling.

Details on data preparation can be found in the provided `cote_divoirre_data_preparation.ipynb` notebook.

We further computed additional features such as vegetation indices as well as interaction features between spectral stats of different months of the year. This feature engineering detail can be found in the provided `Catt_byte_sized_challenge_image_satelite_modelling.ipynb` notebook.

## Modelling and prediction

Based on the data and feature engineering provided in the previous section, we trained three GBDT models, namely LightGBM, Catboost and XGBoost.

Catboost was trained with default parameters, whereas Optuna optimized parameters were used for LightGBM and XGBoost.

The final prediction is a simple average of the individual predictions of the GBDT models.

## Reproducing our solution
- Run the `cote_divoirre_data_preparation.ipynb` notebook to process the provided input data. This will generated two files: `train_monthly_stats_patch_8.csv` and `test_monthly_stats_patch_8.csv`. We have also provided these files in the zipped archive for your convinience.
- Run the provided `Catt_byte_sized_challenge_image_satelite_modelling.ipynb` to fit models and produce the submission file.


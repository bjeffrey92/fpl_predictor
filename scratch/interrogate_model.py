"""
sha-672c043 when this script was run and the conclusion was that the model was not
learning much from own goals, penalties saved and missed, and red cards because they are
rare events
"""

import polars as pl
import shap

from fpl_predictor.model_training import load_23_24_season_data, xgboost

# run model using different number of prediction weeks with 23/24 season data
results = {
    i: xgboost.main(n_prediction_weeks=i, load_data=load_23_24_season_data.load_data)
    for i in range(1, 6)
}

# get the best model and training data for the model
min_loss = min([i[0] for i in results.values()])
best_prediction_weeks = {v[0]: k for k, v in results.items()}[min_loss]
model = results[best_prediction_weeks][1]
data = load_23_24_season_data.load_data(best_prediction_weeks)

# compute shap values for each variable in the model and print in order of importance
explainer = shap.Explainer(model)
shap_values = explainer(data.train_X)
mean_shap_values = abs(shap_values.values.mean(axis=0))
mean_shap_values = {j: i for i, j in zip(mean_shap_values, data.train_X.columns)}
mean_shap_values_df_t = pl.DataFrame(mean_shap_values).transpose()
mean_shap_values_df = mean_shap_values_df_t.with_columns(
    pl.Series("variable", data.train_X.columns)
)
mean_shap_values_df.columns = ["score", "variable"]

# own goals, penalties saved and penalties missed, and red cards all have mean shap values of 0
# probably because they are rare events and so the model doesn't learn much from them
for i in mean_shap_values_df.sort("score").iter_rows():
    print(i)

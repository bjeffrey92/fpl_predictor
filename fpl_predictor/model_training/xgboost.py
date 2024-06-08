from functools import partial
from typing import Callable

import polars as pl
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error

from fpl_predictor.model_training.load_23_24_season_data import TrainTestValData
from fpl_predictor.model_training.load_23_24_season_data import (
    load_data as load_23_24_data,
)


def train(X: pl.DataFrame, y: pl.Series, **kwargs) -> xgb.XGBRegressor:
    if "n_estimators" in kwargs:
        kwargs["n_estimators"] = int(kwargs["n_estimators"])
    if "max_depth" in kwargs:
        kwargs["max_depth"] = int(kwargs["max_depth"])
    model = xgb.XGBRegressor(**kwargs, n_jobs=-1, random_state=1)
    model.fit(X, y)
    return model


def train_and_evaluate(
    train_X: pl.DataFrame,
    train_y: pl.Series,
    val_X: pl.DataFrame,
    val_y: pl.Series,
    **kwargs,
) -> float:
    model = train(train_X, train_y, **kwargs)
    val_predictions = model.predict(val_X)
    return -mean_squared_error(val_y, val_predictions)


def optimise_hyperparameters(
    train_X: pl.DataFrame,
    train_y: pl.Series,
    val_X: pl.DataFrame,
    val_y: pl.Series,
    init_points: int = 25,
    n_iter: int = 50,
) -> xgb.XGBRegressor:
    pbounds = {
        "n_estimators": (100, 1000),
        "max_depth": (2, 25),
        "min_child_weight": (1, 20),
        "max_delta_step": (0, 25),
        "learning_rate": (0.001, 0.5),
        "gamma": (0, 10),
        "reg_alpha": (0, 10),
        "reg_lambda": (0, 20),
    }
    f = partial(
        train_and_evaluate, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y
    )
    optimizer = BayesianOptimization(
        f=f,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    best_model = train(train_X, train_y, **optimizer.max["params"])
    return best_model


def main(  # pragma: no cover
    n_prediction_weeks: int = 2,
    load_data: Callable[[int], TrainTestValData] = load_23_24_data,
) -> tuple[float, xgb.XGBRegressor]:
    data = load_data(n_prediction_weeks)
    model = optimise_hyperparameters(data.train_X, data.train_y, data.val_X, data.val_y)
    test_preds = model.predict(data.test_X)
    mse = mean_squared_error(data.test_y, test_preds)
    return mse, model

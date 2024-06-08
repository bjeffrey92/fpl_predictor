import polars as pl
import pytest
import xgboost as xgb
from sklearn.datasets import load_diabetes

from fpl_predictor.model_training import xgboost


@pytest.fixture
def test_data() -> dict[str, pl.DataFrame | pl.Series]:
    diabetes = load_diabetes()
    X = pl.DataFrame(diabetes.data)
    y = pl.Series(diabetes.target)
    X_train, y_train = X.clone().head(20), y.clone().head(20)
    X_val, y_val = X.clone().tail(10), y.clone().tail(10)
    return {"train_X": X_train, "train_y": y_train, "val_X": X_val, "val_y": y_val}


def test_train_and_evaluate(test_data: dict[str, pl.DataFrame | pl.Series]) -> None:
    mse = xgboost.train_and_evaluate(**test_data, n_estimators=100.1, max_depth=3.0)  # type: ignore[arg-type]
    assert isinstance(mse, float)


def test_optimise_hyperparameters(test_data: tuple[pl.DataFrame, pl.Series]) -> None:
    model = xgboost.optimise_hyperparameters(**test_data, init_points=1, n_iter=1)  # type: ignore[arg-type]
    assert isinstance(model, xgb.XGBRegressor)

import xgboost as xgb

from src.modeling import load_model, save_model, split_data, train_model


def test_split_data(mock_feature_data):
    """Tests that the data is split correctly according to the dates."""
    X, y = mock_feature_data
    split_dates = {
        'train_end': '2022-12-31 23:00:00',
        'val_start': '2023-01-01 00:00:00',
        'val_end': '2023-12-31 23:00:00',
        'test_start': '2024-01-01 00:00:00'
    }

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, split_dates)

    assert not X_train.empty
    assert not X_val.empty
    assert not X_test.empty
    assert X_train.index.get_level_values('datetime').max().year == 2022
    assert X_val.index.get_level_values('datetime').min().year == 2023
    assert X_test.index.get_level_values('datetime').min().year == 2024

def test_train_model(mock_feature_data):
    """Tests that the train_model function runs and returns a trained model."""
    X, y = mock_feature_data
    X_train, y_train, _, _, _, _ = split_data(X, y, {
        'train_end': '2022-12-31 23:00:00',
        'val_start': '2023-01-01 00:00:00',
        'val_end': '2023-12-31 23:00:00',
        'test_start': '2024-01-01 00:00:00'
    })

    estimator_params = {'objective': 'reg:squarederror', 'random_state': 42}
    search_params = {
        'n_iterations': 1,
        'n_cv_splits': 2,
        'scoring': 'neg_root_mean_squared_error',
        'verbose': 0,
        'random_state': 42,
        'param_distributions': {'n_estimators': "randint(10, 20)"}
    }

    _, best_params = train_model(X_train, y_train, estimator_params, search_params)

    assert best_params is not None
    assert 'n_estimators' in best_params

def test_save_and_load_model(tmp_path, mock_feature_data):
    """Tests that a model can be saved and loaded correctly."""
    X, y = mock_feature_data
    model_dir = tmp_path / "models"
    model_filename = "test_model.joblib"

    # Create a simple trained model
    simple_model = xgb.XGBRegressor()
    simple_model.fit(X.iloc[:10], y.iloc[:10])

    # Save and load
    save_model(simple_model, str(model_dir), model_filename)
    loaded_model = load_model(str(model_dir), model_filename)

    assert loaded_model is not None
    assert isinstance(loaded_model, xgb.XGBRegressor)

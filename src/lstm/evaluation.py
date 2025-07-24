import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

def evaluate_lstm_model(model, X_test_seq, y_test_seq, target_scaler):
    """
    Evaluates the trained LSTM model on the test set and returns performance metrics.
    """    
    y_pred_scaled = model.predict(X_test_seq)
    y_pred_unscaled = target_scaler.inverse_transform(y_pred_scaled).flatten()
    y_test_unscaled = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    
    if np.isnan(y_pred_unscaled).any() or np.isinf(y_pred_unscaled).any():
        logging.error("NaNs or Infs detected in predictions after inverse transform. Model may have diverged.")
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan}

    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    r2 = r2_score(y_test_unscaled, y_pred_unscaled)

    logging.info("LSTM Model Performance:")
    logging.info("  MAE:  %.3f MW", mae)
    logging.info("  RMSE: %.3f MW", rmse)
    logging.info("  R²:   %.3f", r2)

    return {'mae': mae, 'rmse': rmse, 'r2': r2}

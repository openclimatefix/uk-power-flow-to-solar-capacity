import logging

import numpy as np
from sklearn.preprocessing import StandardScaler


def create_scalers(X_train, y_train):
    """
    Fits StandardScaler objects on the training data for features and target.
    """
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit scalers only on training data
    feature_scaler.fit(X_train.droplevel('site_id'))
    target_scaler.fit(y_train.droplevel('site_id').values.reshape(-1, 1))

    logging.info("Feature and target scalers fitted on training data.")
    return feature_scaler, target_scaler

def create_sequences(X_df, y_series, look_back, feature_scaler, target_scaler):
    """
    Transforms scaled data into sequences for LSTM training or prediction.
    """
    X_sequences, y_labels = [], []

    # Sort - ensure correct sequence creation for each site
    X_df_sorted = X_df.sort_index(level=['site_id', 'datetime'])
    y_series_sorted = y_series.sort_index(level=['site_id', 'datetime'])

    site_ids = X_df_sorted.index.get_level_values('site_id').unique()

    for site_id in site_ids:
        X_site = X_df_sorted.loc[site_id]
        y_site = y_series_sorted.loc[site_id]

        if len(X_site) < look_back + 1:
            logging.warning(
                "Site %s has %d samples, less than look_back+1 (%d). Skipping.",
                site_id, len(X_site), look_back + 1
            )
            continue

        X_scaled = feature_scaler.transform(X_site)
        y_scaled = target_scaler.transform(y_site.values.reshape(-1, 1)).flatten()

        for i in range(len(X_scaled) - look_back):
            X_sequences.append(X_scaled[i : i + look_back])
            y_labels.append(y_scaled[i + look_back])

    if not X_sequences:
        logging.error("No sequences were created. Check data length and look_back window.")
        return np.array([]), np.array([])

    return np.array(X_sequences), np.array(y_labels)

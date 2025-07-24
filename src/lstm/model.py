from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_lstm_model(look_back_window, n_features, model_config):
    """
    Builds and compiles LSTM model architecture.
    """
    model = Sequential([
        LSTM(
            units=model_config['lstm_units_1'],
            activation=model_config['activation'],
            input_shape=(look_back_window, n_features),
            return_sequences=True
        ),
        Dropout(model_config['dropout_rate']),
        LSTM(
            units=model_config['lstm_units_2'],
            activation=model_config['activation'],
            return_sequences=False
        ),
        Dropout(model_config['dropout_rate']),
        Dense(units=1)
    ])

    optimizer = Adam(
        learning_rate=model_config['optimizer']['learning_rate'],
        clipnorm=model_config['optimizer']['clipnorm']
    )

    model.compile(
        optimizer=optimizer,
        loss=model_config['loss_function']
    )

    model.summary()
    return model

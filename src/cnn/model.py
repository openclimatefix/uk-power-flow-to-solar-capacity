from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_cnn_model(look_back_window, n_features, model_config):
    """
    Builds / compiles 1D CNN model.
    """
    model = Sequential([
        Conv1D(
            filters=model_config['filters_1'],
            kernel_size=model_config['kernel_size'],
            activation=model_config['activation'],
            input_shape=(look_back_window, n_features)
        ),
        MaxPooling1D(pool_size=model_config.get('pool_size', 2)),
        Dropout(model_config['dropout_rate']),
        Conv1D(
            filters=model_config['filters_2'],
            kernel_size=model_config['kernel_size'],
            activation=model_config['activation']
        ),
        MaxPooling1D(pool_size=model_config.get('pool_size', 2)),
        Dropout(model_config['dropout_rate']),
        Flatten(),
        Dense(units=model_config['dense_units'], activation=model_config['activation']),
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

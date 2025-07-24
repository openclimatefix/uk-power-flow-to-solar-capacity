from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def train_cnn_model(model, X_train_seq, y_train_seq, training_config):
    """
    Trains 1D CNN model with early stopping and learning rate reduction.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=training_config['early_stopping_patience'],
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=training_config['reduce_lr_factor'],
        patience=training_config['reduce_lr_patience'],
        min_lr=training_config['min_learning_rate'],
        verbose=1
    )

    history = model.fit(
        X_train_seq,
        y_train_seq,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        validation_split=training_config['validation_split'],
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    return model, history

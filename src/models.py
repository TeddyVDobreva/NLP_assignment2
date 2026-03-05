import tensorflow as tf
import keras

def CNN(X_train, X_validation, X_test, y_train, y_validation, y_test, vocab_size, embedding_dim):
    X_train = X_train.toarray()
    X_validation = X_validation.toarray()
    X_test = X_test.toarray()

    X_train[..., None]
    X_validation = X_validation[..., None]
    X_test = X_test[..., None]

    num_features = X_train.shape[1]
    num_classes = len(set(y_train))

    model = keras.Sequential(
        [
            keras.layers.Embedding(
                vocab_size,  # vocab size pls add
                embedding_dim,
                mask_zero=True, #change mask to pad (like in tut)
            ),
            keras.layers.Dropout(rate=0.25),
            keras.layers.Conv1D(
                filters=num_classes,
                kernel_size=5,
                activation="relu",
                input_shape=(num_features, 1),
            ),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dense(
                num_classes, activation="softmax"
            ), 
            keras.layers.Dropout(rate=0.25),
        ]
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=10,
        batch_size=30,
    )

    model.evaluate(X_test, y_test)
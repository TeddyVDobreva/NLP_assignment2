import keras

class CNN:
    def __init__(self, vocab_size, embedding_dim):
        self.model = keras.Sequential(
            [
                keras.layers.Embedding(
                    vocab_size,  # vocab size pls add
                    embedding_dim,
                    mask_zero=True, #change mask to pad (like in tut)
                ),
                keras.layers.Dropout(rate=0.25),
                keras.layers.Conv1D(
                    filters=4,
                    kernel_size=5,
                    activation="relu",
                ),
                keras.layers.GlobalMaxPooling1D(),
                keras.layers.Dense(
                    4, activation="softmax"
                ), 
                keras.layers.Dropout(rate=0.25),
            ]
        )
        
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        
    def fit(self, X_train, X_validation, epochs=10, batch_size=30):
        self.model.fit(
        X_train,
        validation_data=X_validation,
        epochs=epochs,
        batch_size=batch_size,
        )
        
    def evaluate(self, X_test):
        self.model.evaluate(X_test)



class LSTM:
    def __init__(self, vocab_size, embedding_dim):
        self.model = keras.Sequential(
            [
                keras.layers.Embedding(
                    vocab_size,  # vocab size pls add
                    embedding_dim,
                    mask_zero=True, #change mask to pad (like in tut)
                ),
                keras.layers.Dropout(rate=0.25),
                keras.layers.LSTM(
                    filters=4,
                    kernel_size=5,
                    activation="relu",
                ),
                keras.layers.GlobalMaxPooling1D(),
                keras.layers.Dense(
                    4, activation="softmax"
                ), 
                keras.layers.Dropout(rate=0.25),
            ]
        )
        
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        
    def fit(self, X_train, X_validation, epochs=10, batch_size=30):
        self.model.fit(
        X_train,
        validation_data=X_validation,
        epochs=epochs,
        batch_size=batch_size,
        )
        
    def evaluate(self, X_test):
        self.model.evaluate(X_test)

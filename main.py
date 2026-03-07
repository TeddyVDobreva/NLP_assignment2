from src.data_handler import get_data
from src.models import CNN

# load data + split + preprocessing
vocab=dict()
(
    X_train,
    X_validation,
    X_test,
    y_train,
    y_validation,
    y_test,
    original_train,
    original_validation,
    original_test,
    vocab,
) = get_data("data")

cnn = CNN(len(vocab), 64)
cnn.fit(X_train, X_validation)

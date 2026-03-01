from src.data_handler import get_data


# load data + split + preprocessing
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
) = get_data("data")

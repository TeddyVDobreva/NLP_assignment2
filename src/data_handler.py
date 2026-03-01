from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def get_data(
    path: str,
) -> tuple[
    Any,  # X_train
    Any,  # X_validation
    Any,  # X_test
    Any,  # y_train
    Any,  # y_validation
    Any,  # y_test
    Any,  # original_train
    Any,  # original_validation
    Any,  # original_test
]:
    """
    Load the raw data from the path and split it into training, validation, and test data.
    Returns the processed features and labels, and the original data.

    Arguments:
        path: str- The path to the dataset.

    Returns:
        Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]
            A tuple which contains:
            - X_train: Processed training data.
            - X_validation: Processed validation data.
            - X_test: Processed test data.
            - y_train: Training labels.
            - y_validation: Validation labels.
            - y_test: Test labels.
            - original_train: Original training data.
            - original_validation: Original validation data.
            - original_test: Original test data.
    """
    training_data, validation_data, test_data, y_train, y_validation, y_test = (
        read_data(path)
    )
    (
        X_train,
        X_validation,
        X_test,
        original_train,
        original_validation,
        original_test,
    ) = preprocess_data(training_data, validation_data, test_data)

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        original_train,
        original_validation,
        original_test,
    )


def read_data(
    path: str,
) -> tuple[
    pd.DataFrame,  # training data
    pd.DataFrame,  # validation data
    pd.DataFrame,  # test data
    pd.Series,  # y_train
    pd.Series,  # y_validation
    pd.Series,  # y_test
]:
    """
    Function reads the data from train.csv and test.csv, and extracts the
    column Class Index as labels; Splits the training data into training
    and validation, then prints the lengths of all data splits.

    Arguments:
        path: str- The path to the dataset.

    Returns:
        Tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]
            A tuple which contains:
            - training_data: Training data.
            - validation_data: Validation data.
            - test_data: Test data.
            - y_train: Training labels.
            - y_validation: Validation labels.
            - y_test: Test labels.
    """
    train_data = pd.read_csv(path + "/train.csv")
    test_data = pd.read_csv(path + "/test.csv")

    labels_train = train_data["Class Index"]
    y_test = test_data["Class Index"]

    training_data, validation_data, y_train, y_validation = train_test_split(
        train_data, labels_train, test_size=0.1, random_state=67
    )

    print(f"length of train: {len(training_data)}")
    print(f"length of labels train: {len(y_train)}")
    print(f"length of validation: {len(validation_data)}")
    print(f"length of labels validation: {len(y_validation)}")
    print(f"length of test: {len(test_data)}")
    print(f"length of test labels: {len(y_test)}")

    return training_data, validation_data, test_data, y_train, y_validation, y_test


def preprocess_data(
    training_data: pd.DataFrame, validation_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[
    Any,  # X_train
    Any,  # X_validation
    Any,  # X_test
    pd.Series,  # original_train
    pd.Series,  # original_validation
    pd.Series,  # original_test
]:
    """
    Function combines the Title an Description columns into one variable, and
    applies a TF- IDF vectorizer (with english stopwords). Then, the vectorizer
    is used to transform the training, validation, and test data sets.

    Arguments:
        training_data: pd.DataFrame- Training data which contains the Title
            and Description columns.
        validation_data: pd.DataFrame- Validation data which contains the Title
            and Description columns.
        test_data: pd.DataFrame- Test data which contains the Title
            and Description columns.
    Returns:
        Tuple[csr_matrix, csr_matrix, csr_matrix, Series, Series, Series]
            A tuple which contains:
            - X_train: TF-IDF training set.
            - X_validation: TF-IDF validation set.
            - X_test: TF-IDF test set.
            - original_train: Training set.
            - original_validation: Validation set.
            - original_test: Test set.
    """
    original_train = training_data["Title"] + training_data["Description"]
    original_validation = validation_data["Title"] + validation_data["Description"]
    original_test = test_data["Title"] + test_data["Description"]

    tfidf = TfidfVectorizer(stop_words="english")
    X_train = tfidf.fit_transform(original_train)
    X_validation = tfidf.transform(original_validation)
    X_test = tfidf.transform(original_test)

    return (
        X_train,
        X_validation,
        X_test,
        original_train,
        original_validation,
        original_test,
    )

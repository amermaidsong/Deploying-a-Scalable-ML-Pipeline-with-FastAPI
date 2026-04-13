import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference


def test_train_model():
    """
    Verify that train_model returns a trained RandomForestClassifier.
    """
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)
   
    assert hasattr(model, "estimators_")


def test_compute_model_metrics():
    """
    Verify that compute_model_metrics returns correct values for a known input.
    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Expected Precision: 2/(2+0) = 1.0
    # Expected Recall: 2/(2+1) = 0.666...
    assert precision == 1.0
    assert recall == pytest.approx(0.666, abs=1e-2)
    assert isinstance(fbeta, float)


def test_inference():
    """
    Verify that inference returns the correct shape and valid binary labels.
    """
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model = train_model(X, y)

    preds = inference(model, X)

    assert preds.shape[0] == 2

    assert np.all((preds == 0) | (preds == 1))

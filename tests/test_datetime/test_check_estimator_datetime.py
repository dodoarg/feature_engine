import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.datetime.classes_to_test import DatetimeFeaturesTestClass


@pytest.mark.parametrize(
    "Estimator",
    [
        DatetimeFeaturesTestClass(),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)

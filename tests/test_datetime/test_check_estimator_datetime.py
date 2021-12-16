import pytest
from sklearn.utils.estimator_checks import check_estimator

from feature_engine.datetime.wrapper_classes_for_check_estimator import (
    DatetimeFeaturesTestClass,
)


@pytest.mark.parametrize(
    "Estimator",
    [
        DatetimeFeaturesTestClass(),
    ],
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)

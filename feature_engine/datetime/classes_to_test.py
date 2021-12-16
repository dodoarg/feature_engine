"""
This module contains classes that wrap the main classes in the module datetime, so
that we can run the check_estimator tests from sklearn.
"""
from typing import List, Union

from feature_engine.datetime.datetime import DatetimeFeatures
from feature_engine.validation import _return_tags


class DatetimeFeaturesTestClass(DatetimeFeatures):
    def __init__(
        self,
        variables: Union[None, int, str, List[Union[str, int]]] = None,
        features_to_extract: Union[None, str, List[str]] = None,
        drop_original: bool = True,
        missing_values: str = "raise",
        dayfirst: bool = False,
        yearfirst: bool = False,
        utc: Union[None, bool] = None,
        ignore_format: bool = True,
    ) -> None:
        super().__init__(
            variables,
            features_to_extract,
            drop_original,
            missing_values,
            dayfirst,
            yearfirst,
            utc,
        )
        self._ignore_format = ignore_format
        self.ignore_format = ignore_format

    def _more_tags(self):
        tags_dict = _return_tags()
        # add additional test that fails
        tags_dict["_xfail_checks"]["check_estimators_nan_inf"] = "transformer allows NA"
        tags_dict["_xfail_checks"][
            "check_no_attributes_set_in_init"
        ] = "transformer allows NA"
        return tags_dict

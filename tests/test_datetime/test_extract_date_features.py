from typing import Type
import pandas as pd
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.variable_manipulation import _convert_variables_to_datetime
from feature_engine.datetime import ExtractDateFeatures

def test_extract_date_features(df_vartypes2):
    original_columns = df_vartypes2.columns
    vars_dt     = ["dob", "doa"]
    vars_non_dt = ["Name", "City", "Age", "Marks"]
    vars_mix    = ["Name", "Age", "doa"]
    df_transformed_full = _convert_variables_to_datetime(df_vartypes2).join(
        pd.DataFrame({
            "dob_month"       : [2,2,2,2],
            "doa_month"       : [12,2,6,5],
            "dob_quarter"     : [1,1,1,1],
            "doa_quarter"     : [4,1,2,2],
            "dob_semester"    : [1,1,1,1],
            "doa_semester"    : [2,1,1,1],
            "dob_year"        : [2020,2020,2020,2020],
            "doa_year"        : [2010,1945,2100,1999],
            "dob_woty"        : [9,9,9,9],
            "doa_woty"        : [48,8,24,20],
            "dob_dotw"        : [1,2,3,4],
            "doa_dotw"        : [3,6,1,1],
            "dob_dotm"        : [24,25,26,27],
            "doa_dotm"        : [1,24,14,17],
            "dob_is_weekend"  : [False, False, False, False],
            "doa_is_weekend"  : [False, True, False, False],
            "dob_wotm"        : [4,4,4,4],
            "doa_wotm"        : [1,4,2,3]
        })
    )

    #check exceptions upon class instantiation
    with pytest.raises(ValueError):
        transformer = ExtractDateFeatures(features_to_extract="not_supported")
    with pytest.raises(ValueError):
        transformer = ExtractDateFeatures(variables = 3.519)

    #check transformer attributes
    transformer = ExtractDateFeatures()
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == None
    assert transformer.features_to_extract == ["year"]
    
    #check exceptions upon calling fit method
    with pytest.raises(TypeError):
        transformer.fit("not_a_df")
    with pytest.raises(TypeError):
        ExtractDateFeatures(variables = vars_mix).fit(df_vartypes2)
    with pytest.raises(ValueError):
        transformer.fit(df_vartypes2[vars_non_dt])
    with pytest.raises(ValueError):
        transformer.fit(pd.DataFrame({
            "dates_na": ["Feb-2010", np.nan, "Jun-1922", np.nan]
        }))
    

    #check exceptions upon calling transform method
    transformer.fit(df_vartypes2)
    with pytest.raises(ValueError):
        transformer.transform(df_vartypes2[vars_dt])
    df_na = df_vartypes2.copy()
    df_na.loc[0,'doa'] = np.nan
    with pytest.raises(ValueError):
        transformer.transform(df_na)
    with pytest.raises(NotFittedError):
        ExtractDateFeatures().transform(df_vartypes2)

    #check default initialized transformer
    transformer = ExtractDateFeatures()
    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["dob", "doa"]
    assert transformer.n_features_in_ == 6

    pd.testing.assert_frame_equal(X, df_transformed_full[
        vars_non_dt + ["dob_year", "doa_year"]
    ])

    #check transformer with specified variables to process
    transformer = ExtractDateFeatures(variables = "doa")
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == "doa"
    assert transformer.features_to_extract == ["year"]

    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["doa"]
    pd.testing.assert_frame_equal(
        X, df_transformed_full[vars_non_dt + ["dob"] + ["doa_year"]]
    )

    #check transformer with specified date features to extract
    transformer = ExtractDateFeatures(features_to_extract=["semester", "week_of_the_year"])
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == None
    assert transformer.features_to_extract == ["semester", "week_of_the_year"]

    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["dob", "doa"]
    pd.testing.assert_frame_equal(
        X, df_transformed_full[vars_non_dt + [
            "dob_semester", "doa_semester",
            "dob_woty", "doa_woty"]]
    )

    #check transformer with all date features to extract
    transformer = ExtractDateFeatures(features_to_extract="all")
    assert isinstance(transformer, ExtractDateFeatures)
    assert transformer.variables == None
    assert transformer.features_to_extract == transformer.supported

    X = transformer.fit_transform(df_vartypes2)
    assert transformer.variables_ == ["dob", "doa"]
    pd.testing.assert_frame_equal(X, df_transformed_full.drop(vars_dt, axis=1))

    #check transformer with option to drop datetime features turned off
    X = ExtractDateFeatures(drop_datetime=False).fit_transform(df_vartypes2)
    pd.testing.assert_frame_equal(X, df_transformed_full[
        list(original_columns) + ["dob_year", "doa_year"]
    ])
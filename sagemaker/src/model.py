"""
TRAINING FUNCTIONS: this file in run in 'script mode' when `.fit` is called
from the notebook. `parse_args` and `train_fn` are called in the
`if __name__ =='__main__'` block.
"""
import argparse
import joblib
import os
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import shap

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper

from sklearn.ensemble import RandomForestClassifier


class DFImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median", fill_value=None):
        self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        self.fitted = False
    
    def fit(self, X, y=None):
        self._feature_names = X.columns
        self.imputer.fit(X)
        self.fitted = True
        return self
    
    def transform(self, X):
        assert self.fitted, "Need to cal .fit(X) function first!"
        return pd.DataFrame(
            self.imputer.transform(
                X[self._feature_names]), 
                columns=self._feature_names, 
                index=X.index
            ).astype(np.float32)
    
    def get_feature_names(self):
        return self._feature_names

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def parse_args(sys_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN_DATA"),
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=os.environ.get("SM_CHANNEL_TEST_DATA")
    )
    args, _ = parser.parse_known_args(sys_args)
    return args


def train_fn(args):
    print("loading data")
    train_df = pd.read_csv(args.train_data + "/train.csv", engine='python')
    test_df = pd.read_csv(args.test_data+ "/test.csv", engine='python')
    
    TARGET = 'SeriousDlqin2yrs'
    X_train = train_df.drop(TARGET, axis=1)
    y_train = train_df[TARGET]
    X_test = test_df.drop(TARGET, axis=1)
    y_test = test_df[TARGET]

    print("Imputing missing values")
    transformer = DataFrameMapper([
        (['MonthlyIncome'], DFImputer(strategy="constant", fill_value=-1)),
        (['age'], DFImputer(strategy="median")),
        (['NumberOfDependents'], DFImputer(strategy="median")),
        (['DebtRatio'], DFImputer(strategy="median")),
        (['RevolvingUtilizationOfUnsecuredLines'], DFImputer(strategy="median")),
        (['NumberRealEstateLoansOrLines'], DFImputer(strategy="median")),
        (['NumberOfOpenCreditLinesAndLoans'], DFImputer(strategy="median")),
        (['NumberOfTime30-59DaysPastDueNotWorse'], DFImputer(strategy="median")),
        (['NumberOfTime60-89DaysPastDueNotWorse'], DFImputer(strategy="median")),
        (['NumberOfTimes90DaysLate'], DFImputer(strategy="median")),   
    ], input_df=True, df_out=True)
    transformer.fit(X_train)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)

    print("Building model...")
    model = RandomForestClassifier(n_estimators=50, max_depth=6, max_leaf_nodes=30)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)

    print("Saving artifacts...")
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)

    joblib.dump(transformer, open(str(model_dir / "transformer.joblib"), "wb"))
    joblib.dump(model, open(str(model_dir / "model.joblib"), "wb"))
    joblib.dump(explainer, open(str(model_dir / "explainer.joblib"), "wb"))


def model_fn(model_dir):
    """loads artifacts from model_dir and bundle them in a model_assets dict"""
    model_dir = Path(model_dir)
    transformer = joblib.load(model_dir / "transformer.joblib")
    model = joblib.load(model_dir / "model.joblib")
    explainer = joblib.load(model_dir / "explainer.joblib")

    explainer = shap.TreeExplainer(model)

    model_assets = {
        "transformer": transformer,
        "model": model,
        "explainer": explainer
    }
    return model_assets


def input_fn(request_body_str, request_content_type):
    """takes input json and returns a request dict with 'data' key"""
    assert request_content_type == "application/json", \
        "content_type must be 'application/json'"

    json_obj = json.loads(request_body_str)
    if isinstance(json_obj, str):
        # sometimes you have to unpack the json string twice for some reason. 
        json_obj = json.loads(json_obj)

    request = {
        'df': pd.DataFrame(json_obj)
    }
    return request


def predict_fn(request, model_assets):
    """
    takes a request dict and model_assets dict and returns a response dict
    with 'prediction', 'shap_base' and 'shap_values'
    """ 
    print(f"data: {request['df']}")
    features = model_assets["transformer"].transform(request['df'])

    preds = model_assets["model"].predict_proba(features)[:, 1]

    expected_value = model_assets["explainer"].expected_value

    if expected_value.shape == (1,):
        expected_value = expected_value[0].tolist()
    else:
        expected_value = expected_value[1].tolist()

    shap_values = np.transpose(model_assets["explainer"].shap_values(features)[1])

    response = {}
    response['prediction'] = preds
    response['shap_base'] = expected_value
    response['shap_values'] = {k: v for k, v in zip(features.columns.tolist(), shap_values.tolist())}
    return response


def output_fn(response, response_content_type):
    """takes a response dict and returns a json string of response"""
    assert (
        response_content_type == "application/json"
    ), "accept must be 'application/json'"
    response_body_str = json.dumps(response, cls=NumpyEncoder)
    return response_body_str


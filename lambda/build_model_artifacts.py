from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn_pandas import DataFrameMapper

import shap
import joblib

from src.imputer import DFImputer

print("Loading data...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

TARGET = 'SeriousDlqin2yrs'
X_train = train_df.drop(TARGET, axis=1)
y_train = train_df[TARGET]
X_test = test_df.drop(TARGET, axis=1)
y_test = test_df[TARGET]


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

print("Building explainer...")
shap_explainer = shap.TreeExplainer(model)

print("Saving artifacts...")
joblib.dump(transformer, open(str(Path.cwd() / "pkl" / "transformer.joblib"), "wb"))
joblib.dump(model, open(str(Path.cwd() / "pkl" / "model.joblib"), "wb"))
joblib.dump(shap_explainer, open(str(Path.cwd() / "pkl" / "shap_explainer.joblib"), "wb"))



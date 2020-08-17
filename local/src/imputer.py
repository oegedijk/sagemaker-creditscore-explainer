import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


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
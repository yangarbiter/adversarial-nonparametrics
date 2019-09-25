
import numpy as np
from sklearn.base import BaseEstimator
from scipy import sparse
import xgboost as xgb

class XGBModel(BaseEstimator):
    def __init__(self, model_path):
        self.model_path = model_path
        print(model_path)

        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def fit(self, X, y):
        pass

    def predict(self, X):
        ori_input = np.copy(X)

        np.clip(X, 0, 1, X)
        input_data = xgb.DMatrix(sparse.csr_matrix(X)) 
        ori_input = xgb.DMatrix(sparse.csr_matrix(ori_input))
        test_predict = np.array(self.model.predict(input_data))
        test_predict = test_predict.astype(int)
        return test_predict
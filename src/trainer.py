import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import parse_version
from sklearn.multioutput import MultiOutputRegressor
from typing import Dict, List, Tuple, Any
import xgboost as xgb
import joblib
from statsmodels.tsa.ardl import model

from src.evaluation import compare_data, plot_feature_importance, getCorr, annotated_heatmap, visualize_predict
from src.tools import main_gene_selection, bulk_norm, create_train_dir


class Trainer:
    def __init__(self, out_path, model="grood", threads=1):
        self.path = out_path
        self.train_path, self.model_path = create_train_dir(out_path)
        self.X_train, self.X_test, self.Y_train, self.Y_test = (None, None, None, None)
        self.model_name = model
        self.threads = threads
        self.model = None

    def test_train_data_split(self, X, Y, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.Y_train, self.Y_test = (
            train_test_split(X, Y, test_size=test_size, random_state=random_state))

    def train(self, norm, params=None):
        if self.model_name == "grood":
            self.model = MultiOutputRegressor(ensemble.GradientBoostingRegressor(**params), n_jobs=self.threads)
        elif self.model_name == "xgrood":
            self.model = MultiOutputRegressor(xgb.XGBRegressor(**params), n_jobs=self.threads)
        elif self.model_name == "multigrood":
            Xy = xgb.DMatrix(self.X_train, self.Y_train)
            results: Dict[str, Dict[str, List[float]]] = {}

            self.model = xgb.train(
                {
                    "tree_method": "hist",
                    "num_target": self.Y_train.shape[1],
                    "multi_strategy": "multi_output_tree",
                    "max_depth": params["max_depth"],
                    "n_jobs": self.threads,
                },
                dtrain=Xy,
                num_boost_round=params["n_estimators"],
                obj=deconv_loss,
                evals=[(Xy, "Train")],
                evals_result=results,
                custom_metric=rmse,
            )
        else:
            raise ValueError(f"Unknown model '{model}' specified")

        self.model.fit(self.X_train, self.Y_train)

        metadata = {"estimators": self.Y_train.columns.tolist(),
                    "model_type": self.model_name,
                    "norm": norm}
        annotated_model = {"metadata": metadata,
                           "model": self.model}
        joblib.dump(annotated_model, f"{self.model_path}Model.pkl")

        self.evaluate(metadata)

        return annotated_model

    def evaluate(self, metadata: dict[str, str | Any]):
        pred = self.model.predict(self.X_test)
        pred = rescale_pred(pred)
        pred = pd.DataFrame(data=pred,
                            columns=['Pred ' + str(x) for x in metadata['estimators']],
                            index=self.X_test.index.tolist())

        mse = mean_squared_error(np.array(self.Y_test), np.array(pred))
        print('------------------------------------')
        print("Total mean squared error: ", str(mse))
        print('------------------------------------')

        masterTable = pd.concat([self.Y_test, pred], axis=1, ignore_index=True)
        groundTruthSamples = self.Y_test.columns.tolist()
        predSamples = pred.columns.tolist()
        masterTable.columns = groundTruthSamples + predSamples
        masterTable.to_csv(self.train_path + 'MasterTable.csv')

        correlation_table = getCorr(masterTable, self.Y_test.columns.tolist())
        correlation_table.to_csv(self.train_path + 'Evaluated_training_result.csv')
        df_plot = correlation_table.transpose()
        annotated_heatmap(df_plot, self.train_path)

        compare_data(self.Y_test, pred, masterTable, self.train_path, 'regression')
        compare_data(self.Y_test, pred, masterTable, self.train_path, 'response')
        compare_data(self.Y_test, pred, masterTable, self.train_path, 'error')
        if self.model_name != "multigrood":
            plot_feature_importance(self.model, metadata["estimators"], self.train_path)
        visualize_predict(pred, self.train_path)


def rescale_pred(pred):
    """
    Rescale a prediction for a sample to sum of 1 only, if sum > 1
    """

    rescaled_pred = pred
    for row in range(pred.shape[0]):

        prop_sum = np.sum(pred[row, :])

        if prop_sum > 1:
            rescaled_pred[row, :] = pred[row, :] * (1 / prop_sum)

    return rescaled_pred


def deconv_loss(predt: np.ndarray, dtrain: xgb.DMatrix):
    y = dtrain.get_label().reshape(predt.shape)

    sum_preds = np.sum(predt, axis=1)
    penalty = np.maximum(0, sum_preds - 1)[:, None]
    grad = 2 * (predt - y) + 2 * penalty

    hess = 2 * np.ones_like(predt)

    return grad, hess


def rmse(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label().reshape(predt.shape)
    v = np.sqrt(np.sum(np.power(y - predt, 2)))

    return "PyRMSE", v

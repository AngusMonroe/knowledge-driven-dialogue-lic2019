import numpy as np
import matplotlib

matplotlib.use('Agg')
import xgboost as xgb


class XGBRegressor:
    def __init__(self, booster='gbtree', base_score=0., colsample_bylevel=1.,
                 colsample_bytree=0.7, gamma=0.2, learning_rate=0.01, max_delta_step=0.,
                 max_depth=6, min_child_weight=1., missing=None, n_estimators=800,
                 nthread=32, objective='reg:linear', reg_alpha=0.05, reg_lambda=1.,
                 reg_lambda_bias=0., seed=0, silent=True, subsample=0.9
                 ):
        self.param = {
            "objective": objective,
            "booster": booster,
            "eta": learning_rate,
            "max_depth": max_depth,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "alpha": reg_alpha,
            "lambda": reg_lambda,
            "lambda_bias": reg_lambda_bias,
            "seed": seed,
            "silent": 1 if silent else 0,
            "nthread": nthread,
            "max_delta_step": max_delta_step,
        }
        self.missing = missing if missing is not None else np.nan
        self.n_estimators = n_estimators
        self.base_score = base_score

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ("%s(booster=\'%s\', base_score=%f, colsample_bylevel=%f, \n"
                "colsample_bytree=%f, gamma=%f, learning_rate=%f, max_delta_step=%f, \n"
                "max_depth=%d, min_child_weight=%f, missing=\'%s\', n_estimators=%d, \n"
                "nthread=%d, objective=\'%s\', reg_alpha=%f, reg_lambda=%f, \n"
                "reg_lambda_bias=%f, seed=%d, silent=%d, subsample=%f)" % (
                    self.__class__.__name__,
                    self.param["booster"],
                    self.base_score,
                    self.param["colsample_bylevel"],
                    self.param["colsample_bytree"],
                    self.param["gamma"],
                    self.param["eta"],
                    self.param["max_delta_step"],
                    self.param["max_depth"],
                    self.param["min_child_weight"],
                    str(self.missing),
                    self.n_estimators,
                    self.param["nthread"],
                    self.param["objective"],
                    self.param["alpha"],
                    self.param["lambda"],
                    self.param["lambda_bias"],
                    self.param["seed"],
                    self.param["silent"],
                    self.param["subsample"],
                ))

    def fit(self, X, y, feature_names=None):
        data = xgb.DMatrix(X, label=y, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score * np.ones(X.shape[0]))
        self.model = xgb.train(self.param, data, self.n_estimators)
        return self

    def predict(self, X, feature_names=None):
        data = xgb.DMatrix(X, missing=self.missing, feature_names=feature_names)
        data.set_base_margin(self.base_score * np.ones(X.shape[0]))
        y_pred = self.model.predict(data)
        return y_pred

    def plot_importance(self):
        ax = xgb.plot_importance(self.model)
        self.save_topn_features()
        return ax

    def save_topn_features(self, fname="XGBRegressor_topn_features.txt", topn=-1):
        ax = xgb.plot_importance(self.model)
        yticklabels = ax.get_yticklabels()[::-1]
        if topn == -1:
            topn = len(yticklabels)
        else:
            topn = min(topn, len(yticklabels))
        with open(fname, "w") as f:
            for i in range(topn):
                f.write("%s\n" % yticklabels[i].get_text())
from model import XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from utils.pkl_util import load_pkl, to_pkl


def evaluate(y_pred, y_test):
    MAE = np.mean(abs(y_pred - y_test))
    print("MAE is: %.4f"% MAE)


def train(opt):
    X = load_pkl(opt.train_data)
    y = load_pkl(opt.train_target_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    del X, y
    xgb_regressor = XGBRegressor()
    model = xgb_regressor.fit(X_train, y_train)
    to_pkl(model, opt.model_save_path)
    y_pred = model.predict(X_test)
    evaluate(y_pred, y_test)
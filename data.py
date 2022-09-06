import pandas as pd
import numpy as np
import pickle
from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
from bayes_opt import BayesianOptimization

from config import CFG, RIDGE_CFG, RF_CFG, XGB_CFG, LGBM_CFG
from utils import get_data, train_model, eval_func, dict_eval_func

if __name__ == "__main__":

  X, y, X_scaler, y_scaler = get_data()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  print("Linear Regression Model")
  model = LinearRegression()
  linreg_model, mse, r2 = train_model(model, X_train, y_train, X_test, y_test)

  print("Ridge Regression Model")
  model = Ridge(alpha=RIDGE_CFG.alpha, solver=RIDGE_CFG.solver)
  ridge_model, mse, r2 = train_model(model, X_train, y_train, X_test, y_test)

  print("Random Forest Model")
  model = RandomForestRegressor(n_estimators=RF_CFG.n_estimators, random_state=RF_CFG.random_state, max_depth=RF_CFG.max_depth)
  rf_model, mse, r2 = train_model(model, X_train, y_train, X_test, y_test)

  print("XGBoost Model")
  xgb_model = xgb.XGBRegressor(n_estimators=XGB_CFG.n_estimators, max_depth=XGB_CFG.max_depth, learning_rate=XGB_CFG.learning_rate, random_state=XGB_CFG.random_state)
  xgb_model, mse, r2 = train_model(xgb_model, X_train, y_train, X_test, y_test)

  # Bayesian Optimization

# # Linear Regression with L2 Regularization
# ridge_model = Ridge(alpha=0.1)
# ridge_model.fit(X_train, y_train)
# y_pred = ridge_model.predict(X_test)
# print(f"Linear Reg with L2 mse = {mean_squared_error(y_test, y_pred)}")
# print(f"Linear Reg with L2 r2 = {r2_score(y_test, y_pred)}")


# rf_model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=200)
# rf_model.fit(X_train, y_train)
# y_pred = rf_model.predict(X_test)
# print(f"randomforest mse = {mean_squared_error(y_test, y_pred, squared=False)}")
# print(f"randomforest r2 = {r2_score(y_test, y_pred)}")

# xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=200, learning_rate=0.1, random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred = xgb_model.predict(X_test)
# print(f"Xgboost mse = {mean_squared_error(y_test, y_pred)}")
# print(f"Xgboost r2 = {r2_score(y_test, y_pred)}")
# print(f"max_depth = {xgb_model.get_params()['max_depth']}, n_estim = {xgb_model.get_params()['n_estimators']}" )
# # %%
# xgb_model = xgb.XGBRegressor(n_estimators=5000, max_depth=5000, learning_rate=0.01, random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred = xgb_model.predict(X_test)
# print(f"Xgboost mse = {mean_squared_error(y_test, y_pred)}")
# print(f"Xgboost r2 = {r2_score(y_test, y_pred)}")
# print(f"max_depth = {xgb_model.get_params()['max_depth']}, n_estim = {xgb_model.get_params()['n_estimators']}" )
# # %%
# def eval_xgb(ngy, wg, gr_tt, case, nlz, tstcase_0, tstcase_1, tstcase_2):
#   X = np.array([ngy, wg, gr_tt, case, nlz, tstcase_0, tstcase_1, tstcase_2])
#   X = X.reshape((1, 8))
#   X = scaler.transform(X)
#   out = xgb_model.predict(X)
#   return out[0]

# Bounded region of parameter space
# eval_model = partial(eval_func, linreg_model, X_scaler)
def eval_model(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1
# eval_model[X[0]]
# pbounds = {'ngy':(0, 15), 'wg':(0, 16), 'gr_tt':(0, 1), 'case':(0, 10), 'nlz':(0, 5), 'tstcase_0':(0, 1), 'tstcase_1':(0, 1), 'tstcase_2':(0, 1)}
pbounds = {'x':(2, 4), 'y':(-3, 3)}
optimizer = BayesianOptimization(
    f = eval_model,
    pbounds = pbounds,
    random_state = 1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)
# print(optimizer.max)
# X_scaler.inverse_transform(optimizer.max['target'])

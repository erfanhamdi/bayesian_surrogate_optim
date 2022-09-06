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
from utils import get_data, train_model, eval_func

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

# Bounded region of parameter space

eval_model = partial(eval_func, xgb_model, X_scaler)
# eval_model[X[0]]
pbounds = {'ngy':(0, 15), 'wg':(0, 16), 'gr_tt':(0, 1), 'case':(0, 10), 'nlz':(0, 5), 'tstcase_0':(0, 1), 'tstcase_1':(0, 1), 'tstcase_2':(0, 1)}
# , 'gr_tt':(0, 1), 'case':(0, 10), 'nlz':(0, 5), 'tstcase_0':(0, 1), 'tstcase_1':(0, 1), 'tstcase_2':(0, 1)}
# pbounds = {'x':(2, 4), 'y':(-3, 3)}
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

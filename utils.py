import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from config import CFG


def get_data():
  data_address = "data/data_sheet_1.csv"
  data = pd.read_csv(data_address)

  train_columns = ['ngy', 'wg', 'gr-tt', 'case', 'nlz', 'tstcase']
  X = data[train_columns]
  target_columns = ['v-avg']
  y = data[target_columns]

  X_case = pd.get_dummies(X['tstcase'], prefix='case')
  X_ohe = pd.concat([X.drop(['tstcase'], axis=1), X_case], axis = 1)

  X_scaler = MinMaxScaler()
  y_scaler = MinMaxScaler()

  X_minmax = X_scaler.fit_transform(np.array(X_ohe))
  y_minmax = y_scaler.fit_transform(np.array(y))
  return X_minmax, y_minmax, X_scaler, y_scaler

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"mse = {mse}")
    print(f"r2 = {r2}")
    return model, mse, r2

def eval_func(model, scaler, ngy, wg, gr_tt, case, nlz, tstcase_0, tstcase_1, tstcase_2):
    X = np.array([ngy, wg, gr_tt, case, nlz, tstcase_0, tstcase_1, tstcase_2])
    X = X.reshape((1, 8))
    X = scaler.transform(X)
    out = model.predict(X)
    return out[0]

def eval_fun_discretizer(model, scaler, ngy, wg, gr_tt, case, nlz, tstcase_0, tstcase_1, tstcase_2):
    tstcase_0 = int(tstcase_0)
    tstcase_1 = int(tstcase_1)
    tstcase_2 = int(tstcase_2)
    ngy = int(ngy)
    nlz = int(nlz)
    wg = int(wg)
    case = int(case)
    out = eval_func(model, scaler, ngy, wg, gr_tt, case, nlz, tstcase_0, tstcase_1, tstcase_2)
    return out
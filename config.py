from sklearn.preprocessing import MinMaxScaler, StandardScaler

class CFG:
    scaler = MinMaxScaler()
    run_key = 'DAE_LIFT_LOSS' #due to long run time prefer to split into sections of 600 epochs
    out_path = '/saved_models/'

class RIDGE_CFG:
    alpha = 10
    solver = 'auto'

class RF_CFG:
    n_estimators = 10
    random_state = 42
    max_depth = 200

class XGB_CFG:
    n_estimators = 100
    max_depth = 200
    learning_rate = 0.1
    random_state = 42

class LGBM_CFG:
    n_estimators = 100
    max_depth = 200
    learning_rate = 0.1
    random_state = 42
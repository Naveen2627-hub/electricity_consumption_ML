import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, roc_curve, r2_score, mean_absolute_error



def model(x_train, y_train, x_test, y_test):

    rfr = RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=42)
    rfr.fit(x_train, y_train)
    y_pred = rfr.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, mae, r2

if __name__ == '__main__':

    x_train = pd.read_csv(r"")
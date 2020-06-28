import pandas as pd
import numpy as np
import create_data as cd

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def random_forest(X_train, X_test, y_train, y_test):
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    print(
        "Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    )


if __name__ == "__main__":
    data = cd.create_data()

    X_train, X_test, y_train, y_test, background = cd.data_prep(data)
    # X_train, X_test, y_train, y_test, background = cd.data_prep(data, fatty_liver=False)
    X_train, X_test = cd.preform_cpca(X_train, X_test, background)
    # X_train, X_test = cd.preform_pca(X_train, X_test)

    random_forest(X_train, X_test, y_train, y_test)

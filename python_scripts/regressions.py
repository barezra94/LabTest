import pandas as pd
import numpy as np
import create_data as cd
import values as vs
import matplotlib.pyplot as plt
import seaborn
import pickle

from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve


def prep_data(data, test_size=0.3, random_state=13):
    # Remove NaN values
    data = cd.remove_columns_and_nan(
        data,
        vs.features_values_female_partly,
        [
            "parkinsonism",
            "alzheimer",
            "asthma",
            "K760",
            "D50*",
            "D70*",
            "# of Illnesses",
            "visit_age",
            "sex",
        ],
    )

    # Split to Train and Test
    x = data.drop(
        columns=[
            "parkinsonism",
            "alzheimer",
            "asthma",
            "K760",
            "D50*",
            "D70*",
            "# of Illnesses",
        ]
    )

    y = data["# of Illnesses"]

    print("Labels:", y)

    print("Total Number: ", data.shape)
    print("Number of Non-Ill: ", data[data["# of Illnesses"] == 0].shape)
    print("Number of Ill: ", data[data["# of Illnesses"] > 0].shape)

    # Split to Train and Test
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # Normalize the Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, x.columns


def RandomForestAlgo(X_train, X_test, y_train, y_test):
    # Run Random Forest Regressor
    # All - Best: 0.17026426209026466, using {'max_depth': 25, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 1200}
    # Female - Best: 0.13662315456272303, using {'max_depth': 25, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 1200}
    # regressor = RandomForestRegressor(
    #     n_estimators=1200, min_samples_split=2, min_samples_leaf=10, max_depth=25
    # )

    # regressor.fit(X_train, y_train)

    # Save the model with pickle
    filename = "random_forest_regression_classification.sav"
    # pickle.dump(regressor, open(filename, "wb"))

    # Load Model
    loaded_model = pickle.load(open(filename, "rb"))
    y_pred = loaded_model.predict(X_test)

    plt.figure()
    plt.scatter(y_test, y_pred)
    plt.ylabel("predicted")
    plt.xlabel("actual")
    plt.plot()

    plt.show()

    print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    print(
        "Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    )

    # n_estimators = [100, 300, 500, 800, 1200]
    # max_depth = [5, 8, 15, 25, 30]
    # min_samples_split = [2, 5, 10, 15, 100]
    # min_samples_leaf = [1, 2, 5, 10]

    # hyperF = dict(
    #     n_estimators=n_estimators,
    #     max_depth=max_depth,
    #     min_samples_split=min_samples_split,
    #     min_samples_leaf=min_samples_leaf,
    # )

    # gridF = GridSearchCV(regressor, hyperF, cv=3, verbose=1, n_jobs=-1)
    # bestF = gridF.fit(X_train, y_train)

    # print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # print("Best F:", bestF)


if __name__ == "__main__":
    # Create the Data
    data_male, data_female, all_data = cd.create_data_new(
        "../research/ukbb_new_tests.csv"
    )
    all_data = cd.calc_num_of_illnesses(all_data)

    print(all_data.columns)

    # Split data into train and test and Normalize the data
    X_train, X_test, y_train, y_test, features = prep_data(all_data)

    # Run Random Forest Classifier
    RandomForestAlgo(X_train, X_test, y_train, y_test)

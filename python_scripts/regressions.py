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

    print("Number of Non-Ill Test: ", (y_train == 0).shape)
    print("Number of Ill Test: ", (y_train == 1).shape)

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
    filename1 = "random_forest_regression_classification_naive.sav"
    filename2 = "random_forest_regression_classification.sav"
    # pickle.dump(regressor, open(filename, "wb"))

    # Load Model
    loaded_model1 = pickle.load(open(filename1, "rb"))
    loaded_model2 = pickle.load(open(filename2, "rb"))

    y_pred1 = loaded_model1.predict(X_test)
    y_pred2 = loaded_model2.predict(X_test)

    # Calculate the regression line
    m1, b1 = np.polyfit(y_test, y_pred1, 1)
    m2, b2 = np.polyfit(y_test, y_pred2, 1)

    # Create a figure
    fig = plt.figure()

    # Create two subplots and unpack the output array immediately
    ax1, ax2 = fig.subplots(1, 2, sharey=True)
    ax1.set_title("Naive")
    ax1.scatter(y_test, y_pred1)
    ax1.plot(y_test, m1 * y_test + b1, color="orange")

    ax2.set_title("Best Values")
    ax2.scatter(y_test, y_pred2)
    ax2.plot(y_test, m2 * y_test + b2, color="orange")

    # plt.scatter(y_test, y_pred)
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    # plt.xlabel("actual")
    plt.plot()

    plt.show()

    print("Data for random_forest_regression_classification_naive:")
    print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred1))
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred1))
    print(
        "Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred1))
    )

    print("Data for random_forest_regression_classification:")
    print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred2))
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred2))
    print(
        "Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred2))
    )


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

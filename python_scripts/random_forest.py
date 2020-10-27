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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve

import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    precision_recall_curve,
    plot_roc_curve,
    confusion_matrix,
    plot_confusion_matrix,
)

from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# def random_forest(X_train, X_test, y_train, y_test):
#     regressor = RandomForestRegressor(n_estimators=100, random_state=0)
#     regressor.fit(X_train, y_train)
#     y_pred = regressor.predict(X_test)

#     print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
#     print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
#     print(
#         "Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred))
#     )
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

    y = np.where((data["# of Illnesses"] > 0), 1, data["# of Illnesses"])

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
    # Run Random Forest Classifier
    # clf = RandomForestClassifier(
    #     random_state=1,
    #     max_depth=25,
    #     min_samples_leaf=2,
    #     min_samples_split=2,
    #     n_estimators=500,
    # )
    # clf.fit(X_train, y_train)

    # Save the model with pickle
    filename = "random_forest_binary_classification.sav"
    # pickle.dump(clf, open(filename, "wb"))

    # Load Model
    loaded_model = pickle.load(open(filename, "rb"))
    y_pred = loaded_model.predict(X_test)
    # print(result)

    # y_pred = clf.predict(X_test)

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

    # gridF = GridSearchCV(clf, hyperF, cv=3, verbose=1, n_jobs=-1)
    # bestF = gridF.fit(X_train, y_train)

    # print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # print("Confustion Matrix:", confusion_matrix(y_test, y_pred))

    ax = plt.gca()
    # rfc_disp = plot_roc_curve(loaded_model, X_test, y_test, ax=ax)
    plot_confusion_matrix(loaded_model, X_test, y_test, ax=ax)

    # value = roc_auc_score(y_test, y_pred)

    # fpr, tpr, _ = roc_curve(y_test, y_pred)
    # label = "ROC AUC: " + str(value)
    # seaborn.lineplot(fpr, tpr, label=label)
    plt.title("ROC Curve Random Forest Binary Classifier")
    plt.show()

    # print("Best F:", bestF)


def SVMAlgorithm(X_train, X_test, y_train, y_test):
    # Run SMV Classifier
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Defualt SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    # Create the Data
    data_male, data_female, all_data = cd.create_data_new(
        "../research/ukbb_new_tests.csv"
    )
    all_data = cd.calc_num_of_illnesses(all_data)

    # Split data into train and test and Normalize the data
    X_train, X_test, y_train, y_test, features = prep_data(all_data)

    # Run Random Forest Classifier
    RandomForestAlgo(X_train, X_test, y_train, y_test)

    # Run SVM Classifier
    # SVMAlgorithm(X_train, X_test, y_train, y_test)

    """
    After running both classifiers these were the results:
     - Random Forest Accuracy: 0.5081071205639206
     - Defualt SVM Accuracy: 0.518170607310872

    """

    # Try Running all classifiers to check which is the best
    # test_all_classifiers(X_train, X_test, y_train, y_test)

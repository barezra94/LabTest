import pandas as pd
import numpy as np
import create_data as cd
import values as vs

from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt
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


def test_all_classifiers(X_train, X_test, y_train, y_test):
    h = 0.02  # step size in the mesh

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print("Current Classifier: ", name)
        ax = plt.subplot(1, len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("Score: ", score)

        # # Plot the decision boundary. For that, we will assign a color to each
        # # point in the mesh [x_min, x_max]x[y_min, y_max].
        # if hasattr(clf, "decision_function"):
        #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        # else:
        #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # # Put the result into a color plot
        # Z = Z.reshape(xx.shape)
        # ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k")
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="k", alpha=0.6,
        )

        # ax.set_xlim(xx.min(), xx.max())
        # ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        i += 1

    plt.tight_layout()
    plt.show()


def RandomForestAlgo(X_train, X_test, y_train, y_test):
    # Run Random Forest Classifier
    clf = RandomForestClassifier(random_state=1)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    n_estimators = [100, 300, 500, 800, 1200]
    max_depth = [5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]

    hyperF = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

    gridF = GridSearchCV(clf, hyperF, cv=3, verbose=1, n_jobs=-1)
    bestF = gridF.fit(X_train, y_train)

    # print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))

    print("Best F:", bestF)


def SVMAlgorithm(X_train, X_test, y_train, y_test):
    # Run SMV Classifier
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Defualt SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    # Create the Data
    data_male, data_female = cd.create_data_new("../research/ukbb_new_tests.csv")
    data_female = cd.calc_num_of_illnesses(data_female)

    # Split data into train and test and Normalize the data
    X_train, X_test, y_train, y_test, features = prep_data(data_female)

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

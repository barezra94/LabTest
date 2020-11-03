import pandas as pd
import numpy as np
import create_data as cd
import values as vs
import matplotlib.pyplot as plt
import seaborn as sns
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
            # "# of Illnesses",
            "visit_age",
            "sex",
            "years_to_death",
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
            # "# of Illnesses",
            "years_to_death",
        ]
    )

    y = np.where((data["years_to_death"] < 0), 0, 1)

    # outcome that we are trying to predict - if the person died or not - binary
    # time to death - regression

    # ההפרש בין גיל הביקור למוות (זמן עד מוות) - לראות התפלגות של המידע הזה
    # Add Class weight to the classifier
    # Run Grid Search for each of the label types

    # print("Total Number: ", data.shape)
    # print("Number of Non-Ill: ", data[data["# of Illnesses"] <= 0].shape)
    # print("Number of Ill: ", data[data["# of Illnesses"] > 0].shape)

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
    print("Loaded Data")
    print("Train set:", X_train.shape)
    print("Test set:", X_test.shape)

    # Run Random Forest Classifier
    # Grid Search 10 - Classifier
    # clf = RandomForestClassifier(
    #     random_state=1,
    #     max_depth=30,
    #     min_samples_leaf=2,
    #     min_samples_split=5,
    #     n_estimators=1200,
    #     class_weight="balanced",
    # )

    # Grid Search 0 - Classifier
    # clf = RandomForestClassifier(
    #     random_state=1,
    #     max_depth=30,
    #     min_samples_leaf=1,
    #     min_samples_split=2,
    #     n_estimators=1200,
    #     class_weight="balanced",
    # )

    # Grid Search 5 - Classifier
    # clf = RandomForestClassifier(
    #     random_state=1,
    #     max_depth=30,
    #     min_samples_leaf=2,
    #     min_samples_split=5,
    #     n_estimators=1200,
    #     class_weight="balanced",
    # )
    # clf.fit(X_train, y_train)

    # Save the model with pickle
    filename = "random_forest_binary_classification_5.sav"
    # pickle.dump(clf, open(filename, "wb"))

    # Load Model
    loaded_model = pickle.load(open(filename, "rb"))
    y_pred = loaded_model.predict(X_test)

    print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Confustion Matrix:", confusion_matrix(y_test, y_pred))

    ax = plt.gca()
    # rfc_disp = plot_roc_curve(loaded_model, X_test, y_test, ax=ax)
    plot_confusion_matrix(loaded_model, X_test, y_test, ax=ax)

    # value = roc_auc_score(y_test, y_pred)

    # fpr, tpr, _ = roc_curve(y_test, y_pred)
    # label = "ROC AUC: " + str(value)
    # sns.lineplot(fpr, tpr, label=label)
    plt.title("Confustion Matrix Random Forest Binary Classifier - Illnesses 5")
    plt.show()


def SVMAlgorithm(X_train, X_test, y_train, y_test):
    # Run SMV Classifier
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Defualt SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))


def death_plot(data):
    data.sort_values(by=["years_to_death"], na_position="last")

    df = data.groupby(["years_to_death"], as_index=False).count().iloc[:, 0:2]
    df = df.sort_values(by=["FID"])

    sns.barplot(x=df["years_to_death"], y=df["FID"])

    print(df)

    plt.show()

    # print(data)


if __name__ == "__main__":
    # Create the Data
    data_male, data_female, all_data = cd.create_data_new(
        "../research/ukbb_new_tests.csv"
    )
    # all_data = cd.calc_num_of_illnesses(all_data)
    all_data = cd.calc_time_of_death(all_data)
    # death_plot(all_data)

    # Get train and test sets
    # X_train = np.loadtxt("X_train_5.csv", delimiter=",")
    # y_train = np.loadtxt("y_train_5.csv", delimiter=",")
    # X_test = np.loadtxt("X_test_5.csv", delimiter=",")
    # y_test = np.loadtxt("y_test_5.csv", delimiter=",")

    # Split data into train and test and Normalize the data
    X_train, X_test, y_train, y_test, features = prep_data(all_data)

    np.savetxt("X_train_death_reg.csv", X_train, delimiter=",")
    np.savetxt("X_test_death_reg.csv", X_test, delimiter=",")
    np.savetxt("y_train_death_reg.csv", y_train, delimiter=",")
    np.savetxt("y_test_death_reg.csv", y_test, delimiter=",")

    # Run Random Forest Classifier
    # RandomForestAlgo(X_train, X_test, y_train, y_test)

    # Run SVM Classifier
    # SVMAlgorithm(X_train, X_test, y_train, y_test)

    # Try Running all classifiers to check which is the best
    # test_all_classifiers(X_train, X_test, y_train, y_test)

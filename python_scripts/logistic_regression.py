import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from contrastive import CPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    plot_precision_recall_curve,
)

import create_data as cd


def logistic_regression(data):
    # Drop the IL data from the dataset
    df = data[data["K760"] != 3]

    print("K76: ", df["K760"].value_counts() / df.shape[0])
    print("D50*:", df["D50*"].value_counts() / df.shape[0])

    x = df.drop(columns=["K760", "D50*"])
    y1 = df["K760"]
    y2 = df["D50*"]

    print(x.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y1, test_size=0.3, random_state=13
    )

    # define hyperparameters
    w = [
        {1: 1000, 2: 100},
        {1: 1000, 2: 10},
        {1: 1000, 2: 1.0},
        {1: 500, 2: 1.0},
        {1: 400, 2: 1.0},
        {1: 300, 2: 1.0},
        {1: 200, 2: 1.0},
        {1: 150, 2: 1.0},
        {1: 100, 2: 1.0},
        {1: 99, 2: 1.0},
        {1: 10, 2: 1.0},
        {1: 0.01, 2: 1.0},
        {1: 0.01, 2: 10},
        {1: 0.01, 2: 100},
        {1: 0.001, 2: 1.0},
        {1: 0.005, 2: 1.0},
        {1: 1.0, 2: 1.0},
        {1: 1.0, 2: 0.1},
        {1: 10, 2: 0.1},
        {1: 100, 2: 0.1},
        {1: 10, 2: 0.01},
        {1: 1.0, 2: 0.01},
        {1: 1.0, 2: 0.001},
        {1: 1.0, 2: 0.005},
        {1: 1.0, 2: 10},
        {1: 1.0, 2: 99},
        {1: 1.0, 2: 100},
        {1: 1.0, 2: 150},
        {1: 1.0, 2: 200},
        {1: 1.0, 2: 300},
        {1: 1.0, 2: 400},
        {1: 1.0, 2: 500},
        {1: 1.0, 2: 1000},
        {1: 10, 2: 1000},
        {1: 100, 2: 1000},
    ]
    hyperparam_grid = {"class_weight": w}
    # crange = np.arange(0.5, 20.0, 0.5)
    # hyperparam_grid = {
    #     "class_weight": w,
    #     "penalty": ["l1", "l2"],
    #     "C": crange,
    #     "fit_intercept": [True, False],
    # }

    # define model - class_weight={1: 0.001, 2: 1.0}
    lg = LogisticRegression(
        random_state=13, class_weight={1: 1, 2: 99}, max_iter=300000
    )

    # grid = GridSearchCV(lg, hyperparam_grid, scoring="roc_auc", n_jobs=-1, refit=True)
    # grid.fit(x, y1)
    # print(f"Best score: {grid.best_score_} with param: {grid.best_params_}")

    # fit it
    lg.fit(X_train, y_train)
    # test
    y_pred = lg.predict(X_test)
    # performance
    print(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")
    print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
    print(f"Area Under Curve: {roc_auc_score(y_test, y_pred)}")
    print(f"Recall score: {recall_score(y_test,y_pred)}")


def logistic_regression_cpca(data):
    dataFrame = data[data["K760"] != 3]

    x = dataFrame.drop(columns=["K760", "D50*"])
    y1 = dataFrame["K760"]
    y2 = dataFrame["D50*"]

    df = data.loc[:, data.columns.difference(["K760", "D50*"])]

    background = df[(data["K760"] == 3) | (data["D50*"] == 3)]
    background = background.values

    X_train, X_test, y_train, y_test = train_test_split(
        x, y2, test_size=0.3, random_state=13
    )

    mdl = CPCA(n_components=len(cd.features))
    projected_data = mdl.fit_transform(
        X_train, background, alpha_selection="manual", alpha_value=1.06
    )

    # Convert to NumPy array so CPCA calculation will work
    test = X_test.to_numpy()
    test_data = mdl.transform(test, alpha_selection="manual", alpha_value=1.06)

    lg = LogisticRegression(random_state=13, class_weight={1: 1, 2: 1}, max_iter=5000)

    lg.fit(projected_data, y_train)

    y_pred = lg.predict(test_data)

    # performance
    con_matrix = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    y_pred_proba = lg.predict_proba(X_test)[:, 1]

    print("##### CPCA #####")
    print(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")
    print(f"Confusion Matrix: \n{con_matrix}")
    print(f"Area Under Curve: {auc}")
    print(f"Recall score: {recall_score(y_test,y_pred)}")

    # average_precision = average_precision_score(y_test, y_pred)
    # disp = plot_precision_recall_curve(lg, X_test, y_test)
    # disp.ax_.set_title(
    #     "CPCA Precision-Recall curve: " "AP={0:0.2f}".format(average_precision)
    # )

    # y_test = y_test - 1
    # fpr, tpr, *_ = roc_curve(y_test, y_pred_proba)
    # plot_roc_curve(fpr, tpr)


def logistic_regression_pca(data):
    dataFrame = data[data["K760"] != 3]

    x = dataFrame.drop(columns=["K760", "D50*"])
    y1 = dataFrame["K760"]
    y2 = dataFrame["D50*"]

    df = data.loc[:, data.columns.difference(["K760", "D50*"])]

    X_train, X_test, y_train, y_test = train_test_split(
        x, y2, test_size=0.3, random_state=13
    )

    pca_data = PCA(n_components=len(cd.features))
    projected_data = pca_data.fit_transform(X_train)

    test_data = pca_data.transform(X_test)

    lg = LogisticRegression(random_state=13, class_weight={1: 1, 2: 1}, max_iter=5000)

    lg.fit(projected_data, y_train)

    y_pred = lg.predict(test_data)

    # performance
    con_matrix = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    y_pred_proba = lg.predict_proba(X_test)[:, 1]

    print("##### Normal PCA #####")
    print(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")
    print(f"Confusion Matrix: \n{con_matrix}")
    print(f"Area Under Curve: {auc}")
    print(f"Recall score: {recall_score(y_test,y_pred)}")
    # average_precision = average_precision_score(y_test, y_pred)
    # disp = plot_precision_recall_curve(lg, X_test, y_test)
    # disp.ax_.set_title(
    #     "Precision-Recall curve: " "AP={0:0.2f}".format(average_precision)
    # )

    # plt.show()

    # y_test = y_test - 1
    # fpr, tpr, *_ = roc_curve(y_test, y_pred_proba)
    # plot_roc_curve(fpr, tpr)


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()


def plot_confustion_matrix(con_matrix):
    class_names = [1, 2]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # create heatmap
    sns.heatmap(pd.DataFrame(con_matrix), annot=True, cmap="YlGnBu", fmt="g")
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title("Confusion matrix", y=1.1)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    plt.show()


if __name__ == "__main__":
    df = cd.create_data()

    logistic_regression_cpca(df)
    logistic_regression_pca(df)

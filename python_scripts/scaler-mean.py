import create_data as cd
import scipy as sp

import math
import numpy
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import values as vs

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    precision_recall_curve,
)

final_df = pd.DataFrame(
    columns=[
        "Total Number of Patients",
        "Illness Name",
        "Number of Ill Females",
        "Best Test Name",
        "PR AUC",
        "Total Mean AUC",
    ]
)


def confustion_matrixes(data, features_list, tested_illness):
    """
    True positive - values not between the range of the specific test and the patient is ill
    False positive - values not between the range of the specific test and patient not ill
    True Negative - value between the range and patient not ill
    False Negative - value between the range of the specific test and the patient is ill
    """

    df = pd.DataFrame(
        columns=[
            "Key",
            "TP",
            "FP",
            "TN",
            "FN",
            "TPR",
            "FPR",
            "AUC-Score",
            "Precision Score",
            "Recall Score",
            "PR AUC",
        ]
    )

    for key in features_list:
        min_value = features_list[key][0]
        max_value = features_list[key][1]

        true_positive = data[
            ((data[key] < min_value) | (data[key] > max_value))
            & (data[tested_illness] == 0)
        ].count()

        true_negative = data[
            (data[key] >= min_value)
            & (data[key] <= max_value)
            & (data[tested_illness] == 1)
        ].count()

        false_positive = data[
            ((data[key] < min_value) | (data[key] > max_value))
            & (data[tested_illness] == 1)
        ].count()

        false_negative = data[
            (data[key] >= min_value)
            & (data[key] <= max_value)
            & (data[tested_illness] == 0)
        ].count()

        true_positive = true_positive[0]
        true_negative = true_negative[0]
        false_positive = false_positive[0]
        false_negative = false_negative[0]

        # precision_score = true_positive / (true_positive + false_positive)
        recall_score = true_positive / (true_positive + false_negative)

        # Remove NaN Values
        new_data = data[data[key].notna()]

        # Standerdize Data
        mean_key = new_data[key].mean()
        std_key = new_data[key].std()

        new_values = (new_data[key] - mean_key) / std_key

        median_diff = abs(new_values)
        # print(median_diff)

        value = roc_auc_score(new_data[tested_illness], median_diff)
        # precision = precision_score(new_data[tested_illness], median_diff)
        # recall = recall_score(new_data[tested_illness], median_diff)

        precision, recall, _ = precision_recall_curve(
            new_data[tested_illness], median_diff
        )

        auc_score = auc(recall, precision)
        # print("PR AUC: ", auc_score)

        # print("Precision: ", 1 - precision)
        # print("Recall: ", 1 - recall)

        df = df.append(
            {
                "Key": key,
                "TP": true_positive,
                "FP": false_positive,
                "TN": true_negative,
                "FN": false_negative,
                "TPR": (true_positive / (true_positive + false_negative)),
                "FPR": (false_positive / (true_negative + false_positive)),
                "AUC-Score": value,
                "Precision Score": precision_score,
                "Recall Score": recall_score,
                "PR AUC": auc_score,
                "median-diff": median_diff,
                "data": new_data[tested_illness],
            },
            ignore_index=True,
        )

    # seaborn.lineplot(recall, precision, label=illness)

    new_df = df.sort_values(by="AUC-Score", ascending=False)
    best_auc = new_df.head()

    df = df.drop(columns=["median-diff", "data"])
    df.to_csv("confusion_matrix_" + tested_illness + ".csv")

    return df, best_auc


def display_mean_data(df, y_true, illness, axs):
    # Standerdize Data
    mean = df.mean(axis=0)
    mean = mean.reshape(1, -1)

    df_diff = sp.spatial.distance.cdist(mean, df)
    median_diff = df_diff.reshape(-1, 1)

    # Normalize diff - the values need to be between 0 and 1
    # mean_key = df_diff.mean()
    # std_key = df_diff.std()

    # new_values = (df_diff - mean_key) / std_key

    # median_diff = abs(new_values)
    # median_diff = median_diff.reshape(-1, 1)

    value = roc_auc_score(y_true, median_diff)

    fpr, tpr, _ = roc_curve(y_true, median_diff)
    label = "Mean AUC of " + illness + ": " + str(value)
    seaborn.lineplot(fpr, tpr, label=label, ax=axs)

    print("Mean AUC: " + str(value))

    return value


if __name__ == "__main__":

    fig, axs = plt.subplots(3, 2)

    for illness, col_index, row_index in [
        ["parkinsonism", 0, 0],
        ["alzheimer", 0, 1],
        ["asthma", 0, 2],
        ["K760", 1, 0],
        ["D50*", 1, 1],
        ["D70*", 1, 2],
    ]:

        print("Current Illness:", illness)
        data_male, data_female = cd.create_data_new("../research/ukbb_new_tests.csv")

        print("All Female Data: ", data_female.shape)
        print("All Male Data: ", data_male.shape)
        print("Ill Female:", data_female[data_female[illness] == 1].shape)

        all_aucs, best_auc = confustion_matrixes(
            data_female, vs.features_values_female, illness
        )

        # Display Data on Plot
        for index, row in best_auc.iterrows():
            fpr, tpr, _ = roc_curve(row["data"], row["median-diff"])
            seaborn.lineplot(
                fpr,
                tpr,
                label=row["Key"] + " AUC: " + str(row["AUC-Score"]),
                ax=axs[row_index, col_index],
            )

        # Remove problematic values and NaN values
        data_female_new = cd.remove_columns_and_nan(
            data_female, vs.features_values_female, illness
        )

        data_female_n = data_female_new.drop(columns=[illness])

        # Normalize the Data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data_female_n)

        mean_auc = display_mean_data(
            normalized_data,
            data_female_new[illness],
            illness,
            axs[row_index, col_index],
        )

        best_row = best_auc.iloc[0]

        print("TP: ", best_row["TP"])
        print("TN: ", best_row["TN"])
        print("FP: ", best_row["FP"])
        print("FN: ", best_row["FN"])

        final_df = final_df.append(
            {
                "Illness Name": illness,
                "Total Number of Patients": data_female.shape[0],
                "Number of Ill Females": data_female[data_female[illness] == 1].shape[
                    0
                ],
                "Best Test Name": best_row["Key"],
                "ROC AUC": best_row["AUC-Score"],
                "PR AUC": best_row["PR AUC"],
                "Total Mean AUC": mean_auc,
            },
            ignore_index=True,
        )

        axs[row_index, col_index].set_title(illness)

    final_df.to_csv("results.csv")
    for ax in axs.flat:
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    plt.legend()
    plt.show()


"""if __name__ == "__main__":
    data_male, data_female = cd.create_data_new("../research/ukbb_new_tests.csv")
    # data = cd.create_data()

    # Add invalid Test Column
    # data_female = cd.add_invalid_column(data_female, vs.features_values_female)
    # data_male = cd.add_invalid_column(data_male, vs.features_values_male)

    print("All Female Data: ", data_female.shape)
    print("All Male Data: ", data_male.shape)

    confustion_data = confustion_matrixes(
        data_female, vs.features_values_female, "alzheimer"
    )

    # Sort rows according to the AUC score
    # confustion_data = confustion_data.sort_values(by="AUC-Score", ascending=False)
    # best_auc = confustion_data.head()

    # Remove problematic values and NaN values
    data_female_new = cd.remove_columns_and_nan(data_female)

    # Join to one dataframe for normalization
    # data = pd.concat([data_female, data_male])

    # Normalize the Data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_female_new)

    # Get the mean of all the data rows
    mean = normalized_data.mean(axis=0)

    # Select all ill patients and all control patients, compute diff
    # and compute distance from overall mean
    ill_female = data_female_new[
        (data_female_new["alzheimer"] == 0) & (data_female_new["sex"] == 2)
    ]
    # ill_male = data[(data["D70*"] == 2) & (data["sex"] == 1)]
    control_female = data_female_new[
        (data_female_new["alzheimer"] == 1) & (data_female_new["sex"] == 2)
    ]
    # control_male = data[(data["D70*"] == 1) & (data["sex"] == 1)]

    print("Ill Female: ", ill_female.shape)
    # print("Ill Male: ", ill_male.shape)

    # Normalize Ill and Control patients data
    normalized_ill_female = scaler.transform(ill_female)
    # normalized_ill_male = scaler.transform(ill_male)
    normalized_control_female = scaler.transform(control_female)
    # normalized_control_male = scaler.transform(control_male)

    mean = mean.reshape(1, -1)

    ill_diff_female = sp.spatial.distance.cdist(mean, normalized_ill_female)
    # ill_diff_male = sp.spatial.distance.cdist(mean, normalized_ill_male)
    control_diff_female = sp.spatial.distance.cdist(mean, normalized_control_female)
    # control_diff_male = sp.spatial.distance.cdist(mean, normalized_control_male)

    # Keep only rows that have at least one test that is not in range and compute diff
    # invalid_range = data_female[data_female["allTestValid"] == 0]

    # print("All Invalid Tests: ", invalid_range.shape)

    # normalized_invalid_range = scaler.transform(invalid_range)
    # invalid_range_diff = sp.spatial.distance.cdist(mean, normalized_invalid_range)

    # Remove values that are high
    ill_diff_female = ill_diff_female[0]
    # ill_diff_male = ill_diff_male[0]

    control_diff_female = control_diff_female[0]
    # control_diff_male = control_diff_male[0]

    # invalid_range_diff = invalid_range_diff[0]

    seaborn.distplot(
        control_diff_female,
        label="control-female",
        # hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )
    seaborn.distplot(
        ill_diff_female,
        label="ill-female",
        # hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )
    # seaborn.distplot(
    #     control_diff_male,
    #     label="control-male",
    #     # hist_kws={"cumulative": True},
    #     kde_kws={"cumulative": True},
    # )
    # seaborn.distplot(
    #     ill_diff_male,
    #     label="ill-male",
    #     # hist_kws={"cumulative": True},
    #     kde_kws={"cumulative": True},
    # )
    # seaborn.distplot(
    #     invalid_range_diff,
    #     label="invalid range",
    #     # hist_kws={"cumulative": True},
    #     kde_kws={"cumulative": True},
    # )

    plt.legend()
    plt.show()
"""


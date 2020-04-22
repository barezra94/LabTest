import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

features = {
    # Red Blood Cells
    "RBC": "red_blood_cell_erythrocyte_count_f30010_0_0",
    "HCT": "haematocrit_percentage_f30030_0_0",
    "HGB": "haemoglobin_concentration_f30020_0_0",
    "MCV": "mean_corpuscular_volume_f30040_0_0",
    "MCH": "mean_corpuscular_haemoglobin_f30050_0_0",
    "MCHC": "mean_corpuscular_haemoglobin_concentration_f30060_0_0",
    # White Blood Cells
    "WBC": "white_blood_cell_leukocyte_count_f30000_0_0",
    "LYM": "lymphocyte_count_f30120_0_0",
    "LYM_precent": "lymphocyte_percentage_f30180_0_0",
    "MONO": "monocyte_count_f30130_0_0",
    "MONO_precent": "monocyte_percentage_f30190_0_0",
    "NEU": "neutrophill_count_f30140_0_0",
    "EOS_precent": "eosinophill_percentage_f30210_0_0",
    "BASO_precent": "basophill_percentage_f30220_0_0",
    # Platelet Count
    "PLT": "platelet_count_f30080_0_0",
    "MPV": "mean_platelet_thrombocyte_volume_f30100_0_0",
    # Uncatagorized
    # "Albumin_blood": "albumin_f30600_0_0",
    # "Alkaline_Phosphatase_blood": "alkaline_phosphatase_f30610_0_0",
    # "Calcium_total_blood": "calcium_f30680_0_0",
    # "Cholesterol_blood": "cholesterol_f30690_0_0",
    # "Creatinin_blood": "creatinine_f30700_0_0",
    # "GammaGT_blood": "gamma_glutamyltransferase_f30730_0_0",
    # "Glucose_blood": "glucose_f30740_0_0",
    # "HDL_cholesterol_blood": "hdl_cholesterol_f30760_0_0",
    # "LDL_cholesterol_calculated_blood": "ldl_direct_f30780_0_0",
    # "Bilirubin_total_blood": "total_bilirubin_f30840_0_0",
    # "Triglycerides_blood": "triglycerides_f30870_0_0",
}

all_features = {
    "w": "creactive_protein_f30710_0_0",  # Is this equal to wide range or high sensitive
    "ab": "phosphate_f30810_0_0",  # What is this
}


def filter_uk_data():
    df_uk = pd.read_csv("../research/blood_test_uk.csv")
    df_uk_added_data = pd.read_csv("../research/ICD10_UK.csv")

    # Create DataSet to return
    wanted_columns = list(features.values())
    wanted_columns.extend(
        ["FID", "age_when_attended_assessment_centre_f21003_0_0", "sex_f31_0_0"]
    )
    df_uk = df_uk.loc[:, wanted_columns]

    # Drop rows that have NaN values in them
    df_uk = df_uk.dropna()

    # Create a DataSet for UK data that has values from both files
    df_uk = df_uk.merge(df_uk_added_data, on="FID")

    # Replace numbers with Label value
    df_uk["K760"].replace(1, "Positive", inplace=True)
    df_uk["K760"].replace(2, "Control", inplace=True)

    return df_uk


def filter_il_data():
    # TODO: Maybe change to read straight from excel file
    df_il = pd.read_csv("../research/blood_test_il.csv")
    df_uk_added_data = pd.read_csv("../research/ICD10_UK.csv")

    # Filter dataSet values
    wanted_columns = list(features.keys())
    wanted_columns.extend(["hospital_patient_id", "age_computed", "gender"])

    df_il = df_il.loc[:, wanted_columns]

    # Replace numbers with Gender value
    df_il["gender"].replace(1, "Male", inplace=True)
    df_il["gender"].replace(2, "Female", inplace=True)

    # Keep only the latest test of the patient
    df_il = df_il.drop_duplicates(subset="hospital_patient_id", keep="last")

    # Drop rows that have NaN in them
    df_il = df_il.dropna()

    for illness in df_uk_added_data.columns[2:]:
        df_il[illness] = "Malram"

    return df_il


def merge_df(df_uk, df_il):

    # Change column names to match df_uk
    all_columns = {
        "hospital_patient_id": "FID",
        "age_computed": "age_when_attended_assessment_centre_f21003_0_0",
        "gender": "sex_f31_0_0",
    }
    all_columns.update(features)
    df_il = df_il.rename(columns=all_columns)

    df_uk = df_uk.drop(columns=["IID", "FID"])

    # Merge datasets
    df_uk = df_uk.append(df_il)
    df_uk = df_uk.reset_index(drop=True)

    return df_uk


if __name__ == "__main__":
    df_uk = filter_uk_data()
    df_il = filter_il_data()
    merged_df = merge_df(df_uk=df_uk, df_il=df_il)

    x = merged_df.loc[:, features.values()].values
    x = StandardScaler().fit_transform(x)  # normalizing the features

    feat_cols = ["feature" + str(i) for i in range(x.shape[1])]
    normalized_data = pd.DataFrame(x, columns=feat_cols)

    pca_fatty_liver = PCA(n_components=2)
    principalComponent_liver = pca_fatty_liver.fit_transform(x)

    principal_liver_Df = pd.DataFrame(
        data=principalComponent_liver,
        columns=["principal component 1", "principal component 2"],
    )

    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel("Principal Component - 1", fontsize=20)
    plt.ylabel("Principal Component - 2", fontsize=20)
    plt.title("Principal Component Analysis of Fatty Liver", fontsize=20)
    targets = [
        "Positive-Female",
        "Positive-Male",
        "Control-Female",
        "Control-Male",
        "Malram-Female",
        "Malram-Male",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for target, color in zip(targets, colors):
        target_gender = target.split("-")

        indicesToKeep = (merged_df["K760"] == target_gender[0]) & (
            merged_df["sex_f31_0_0"] == target_gender[1]
        )

        plt.scatter(
            principal_liver_Df.loc[indicesToKeep, "principal component 1"],
            principal_liver_Df.loc[indicesToKeep, "principal component 2"],
            facecolors="none",
            edgecolors=color,
        )

    plt.legend(targets, prop={"size": 15})

    plt.show()

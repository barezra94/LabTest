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
    "MONO_percent": "monocyte_percentage_f30190_0_0",
    "NEU": "neutrophill_count_f30140_0_0",
    "EOS_percent": "eosinophill_percentage_f30210_0_0",
    "BASO_percent": "basophill_percentage_f30220_0_0",
    # Platelet Count
    "PLT": "platelet_count_f30080_0_0",
    "MPV": "mean_platelet_thrombocyte_volume_f30100_0_0",
    # Uncatagorized
    "Albumin_blood": "albumin_f30600_0_0",
    "Alkaline_Phosphatase_blood": "alkaline_phosphatase_f30610_0_0",
    "Calcium_total_blood": "calcium_f30680_0_0",
    "Cholesterol_blood": "cholesterol_f30690_0_0",
    "Creatinin_blood": "creatinine_f30700_0_0",
    "GammaGT_blood": "gamma_glutamyltransferase_f30730_0_0",
    "Glucose_blood": "glucose_f30740_0_0",
    "HDL_cholesterol_blood": "hdl_cholesterol_f30760_0_0",
    "LDL_cholesterol_calculated_blood": "ldl_direct_f30780_0_0",
    "Bilirubin_total_blood": "total_bilirubin_f30840_0_0",
    "Triglycerides_blood": "triglycerides_f30870_0_0",
}

all_features = {
    "w": "creactive_protein_f30710_0_0",  # Is this equal to wide range or high sensitive
    "ab": "phosphate_f30810_0_0",  # What is this
}

df_uk = pd.read_csv("../research/blood_test_uk.csv")
df_uk_added_data = pd.read_csv("../research/ICD10_UK.csv")

# Create a DataSet for UK data that has values from both files
df_uk = df_uk.merge(df_uk_added_data, on="FID")

# Replace numbers with Label value
df_uk["K760"].replace(1, "Positive", inplace=True)
df_uk["K760"].replace(2, "Control", inplace=True)

# Replace NaN values with 0 for calculation - is this the right thing to do?
df_uk = df_uk.fillna(0)

x = df_uk.loc[:, features.values()].values
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
targets = ["Positive", "Control"]
colors = ["r", "g"]
for target, color in zip(targets, colors):
    indicesToKeep = df_uk["K760"] == target
    plt.scatter(
        principal_liver_Df.loc[indicesToKeep, "principal component 1"],
        principal_liver_Df.loc[indicesToKeep, "principal component 2"],
        c=color,
        s=50,
    )

plt.legend(targets, prop={"size": 15})

plt.show()

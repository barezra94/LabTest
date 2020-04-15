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
    # Platelet Count
    "PLT": "platelet_count_f30080_0_0",
}

# 'RDW': 'mean_corpuscular_volume_f30040_0_0' ??

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

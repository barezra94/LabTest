import pandas as pd
from scipy import stats
import math

test_map = {
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

df_uk = pd.read_csv("blood_test_uk.csv")
df_il = pd.read_csv("blood_test_il.csv")

for t in test_map:
    """ 
    Comparision Steps:
        1. Get data from both files
        2. Split data in each file to the following two datasets: Male and Female
        3. For Each Test that we want to preform:
            3.1. Remove empty cells from each dataset
            3.2. Preform preform MEAN, STD,   
    """

    mean_il = df_il[t].mean()
    mean_uk = df_uk[test_map[t]].mean()

    std_il = df_il[t].std()
    std_uk = df_uk[test_map[t]].std()

    smd = abs((mean_il - mean_uk)) / math.sqrt(std_il * std_il + std_uk * std_uk)

    df_il[t] = (df_il[t] - df_il[t].mean()) / df_il[t].std()
    df_uk[test_map[t]] = (df_uk[test_map[t]] - df_uk[test_map[t]].mean()) / df_uk[
        test_map[t]
    ].std()

    test = stats.ks_2samp(df_il[t], df_uk[test_map[t]])
    # print(t, mean_il, mean_uk, 'ks reject:', test[1] < 0.05)
    print(
        t,
        "mean_il:",
        mean_il,
        "mean_uk:",
        mean_uk,
        "std_il:",
        std_il,
        "std_uk:",
        std_uk,
    )

print(df_il)

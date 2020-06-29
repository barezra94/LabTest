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
    "wide_range_CRP": "creactive_protein_f30710_0_0",
    "Phosphorus_blood": "phosphate_f30810_0_0",
    "age_computed": "age_when_attended_assessment_centre_f21003_0_0",
    "gender": "sex_f31_0_0",
}
# Ranges are from https://www.wikiwand.com/en/Reference_ranges_for_blood_tests
features_values = {
    # Red Blood Cells
    "red_blood_cell_erythrocyte_count_f30010_0_0": (4.7, 6.1),
    "haematocrit_percentage_f30030_0_0": (42, 53),
    "haemoglobin_concentration_f30020_0_0": (11.7, 15.7),
    "mean_corpuscular_volume_f30040_0_0": (80.8, 100),
    "mean_corpuscular_haemoglobin_f30050_0_0": (26, 32),
    "mean_corpuscular_haemoglobin_concentration_f30060_0_0": (32, 36),
    # White Blood Cells
    "white_blood_cell_leukocyte_count_f30000_0_0": (4, 10),
    "lymphocyte_count_f30120_0_0": (0.7, 3.9),
    "lymphocyte_percentage_f30180_0_0": (16, 33),
    "monocyte_count_f30130_0_0": (0.1, 0.8),
    "monocyte_percentage_f30190_0_0": (3, 7),
    "neutrophill_count_f30140_0_0": (1.8, 7),
    "eosinophill_percentage_f30210_0_0": (1, 7),
    "basophill_percentage_f30220_0_0": (0, 2),
    # Platelet Count
    "platelet_count_f30080_0_0": (140, 450),
    "mean_platelet_thrombocyte_volume_f30100_0_0": (7.2, 11.7),
    # Uncatagorized
    "albumin_f30600_0_0": (35, 55),
    "alkaline_phosphatase_f30610_0_0": (42, 128),  # lower range female, higher male
    "calcium_f30680_0_0": (2.1, 2.7),
    "cholesterol_f30690_0_0": (3.6, 6.5),
    "creatinine_f30700_0_0": (68, 118),
    "gamma_glutamyltransferase_f30730_0_0": (0, 0.92),
    "glucose_f30740_0_0": (3.3, 5.6),
    "hdl_cholesterol_f30760_0_0": (0.9, 2.2),
    "ldl_direct_f30780_0_0": (2.0, 3.4),
    "total_bilirubin_f30840_0_0": (1.7, 22),
    "triglycerides_f30870_0_0": (0.77, 1.7),
    "creactive_protein_f30710_0_0": (0, 6),
    "phosphate_f30810_0_0": (0.8, 1.5),
}

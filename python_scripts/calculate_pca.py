import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
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

pca_data = []
merged_df = []


def filter_uk_data():
    df_uk = pd.read_csv("../research/blood_test_uk.csv")
    df_uk_added_data = pd.read_csv("../research/ICD10_UK.csv")

    # Create DataSet to return
    wanted_columns = list(features.values())
    wanted_columns.extend(["FID"])
    df_uk = df_uk.loc[:, wanted_columns]

    # Drop rows that have NaN values in them
    df_uk = df_uk.dropna()

    # Create a DataSet for UK data that has values from both files
    df_uk = df_uk.merge(df_uk_added_data, on="FID")

    df_uk["sex_f31_0_0"].replace("Male", 1, inplace=True)
    df_uk["sex_f31_0_0"].replace("Female", 2, inplace=True)

    return df_uk


def filter_il_data():
    # TODO: Maybe change to read straight from excel file
    df_il = pd.read_csv("../research/blood_test_il.csv")
    df_uk_added_data = pd.read_csv("../research/ICD10_UK.csv")

    # Filter dataSet values
    wanted_columns = list(features.keys())
    wanted_columns.extend(["hospital_patient_id"])

    df_il = df_il.loc[:, wanted_columns]

    # Keep only the latest test of the patient
    df_il = df_il.drop_duplicates(subset="hospital_patient_id", keep="last")

    # Drop rows that have NaN in them
    df_il = df_il.dropna()

    for illness in df_uk_added_data.columns[2:]:
        df_il[illness] = 3

    return df_il


def merge_df(df_uk, df_il):

    # Change column names to match df_uk
    all_columns = {
        "hospital_patient_id": "FID",
    }
    all_columns.update(features)
    df_il = df_il.rename(columns=all_columns)

    df_uk = df_uk.drop(columns=["IID", "FID"])
    df_il = df_il.drop(columns=["FID"])

    # Merge datasets
    df_uk = df_uk.append(df_il)
    df_uk = df_uk.reset_index(drop=True)

    return df_uk


def calculate_pca(dataFrame, n_components=4):
    x = dataFrame.loc[:, features.values()].values
    x = StandardScaler().fit_transform(x)  # normalizing the features

    feat_cols = ["feature" + str(i) for i in range(x.shape[1])]
    normalized_data = pd.DataFrame(x, columns=feat_cols)

    pca_data = PCA(n_components=n_components)
    principalComponent = pca_data.fit_transform(x)

    print(
        "Explained variation per principal component: {}".format(
            pca_data.explained_variance_ratio_
        )
    )

    principal_data = pd.DataFrame(
        data=principalComponent,
        columns=["principal component " + str(i + 1) for i in range(n_components)],
    )

    return principal_data


def create_axis(dataSet, pca_data, pca_components, illness, count=1000):
    axis = []

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

        indicesToKeep = (dataSet[illness] == target_gender[0]) & (
            dataSet["sex_f31_0_0"] == target_gender[1]
        )

        # place = 0
        # counter = 0
        # for i in range(indicesToKeep.shape[0]):
        #     if indicesToKeep[i] == True and counter <= count:
        #         place = i
        #         counter += 1

        trace = go.Scattergl(
            # x=pca_data[:place].loc[indicesToKeep[:place], pca_components[0]],
            # y=pca_data[:place].loc[indicesToKeep[:place], pca_components[1]],
            x=pca_data.loc[indicesToKeep, pca_components[0]],
            y=pca_data.loc[indicesToKeep, pca_components[1]],
            mode="markers",
            name=target,
            marker=dict(color=color),
        )

        axis.append(trace)

    return axis


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def display_charts(pca_data_columns, options):
    # Create the Column Options to pick from - PCA
    pca_options = []
    for col in pca_data_columns:
        pca_options.append({"label": col, "value": col})

    # Create the Dropdown options - Illnesses
    illness_options = []
    for illness in options:
        illness_options.append({"label": illness, "value": illness})

    app.layout = html.Div(
        [
            html.Div(
                [
                    dcc.Dropdown(id="illness", options=illness_options, value="K760"),
                    dcc.Checklist(
                        id="principle-components",
                        options=pca_options,
                        value=["principal component 1", "principal component 2"],
                        labelStyle={"display": "inline-block"},
                    ),
                ],
                style={"width": "49%", "display": "inline-block"},
            ),
            html.Div([dcc.Graph(id="graph")]),
        ]
    )

    app.run_server(debug=True)


@app.callback(
    dash.dependencies.Output("graph", "figure"),
    [
        dash.dependencies.Input("illness", "value"),
        dash.dependencies.Input("principle-components", "value"),
    ],
)
def change_graph_data(illness, principle_components):
    data = create_axis(merged_df, pca_data, principle_components, illness)

    return {
        "data": data,
    }


if __name__ == "__main__":
    df_uk = filter_uk_data()
    df_il = filter_il_data()
    merged_df = merge_df(df_uk=df_uk, df_il=df_il)

    pca_data = calculate_pca(merged_df)

    # Replace numbers with Gender value
    merged_df["sex_f31_0_0"].replace(1, "Male", inplace=True)
    merged_df["sex_f31_0_0"].replace(2, "Female", inplace=True)

    # Replace numbers with Label value
    for illness in ["K760", "D500", "D501", "D508", "D509"]:
        merged_df[illness].replace(1, "Positive", inplace=True)
        merged_df[illness].replace(2, "Control", inplace=True)
        merged_df[illness].replace(3, "Malram", inplace=True)

    display_charts(pca_data.columns, ["K760", "D500", "D501", "D508", "D509"])

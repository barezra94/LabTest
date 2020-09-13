import pandas as pd
from contrastive import CPCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import values as values


def __filter_uk_data():
    df_uk = pd.read_csv("../research/blood_test_uk.csv")
    df_uk_added_data = pd.read_csv("../research/ICD10_UK.csv")

    # Create DataSet to return
    wanted_columns = list(values.features.values())
    wanted_columns.extend(["FID"])
    df_uk = df_uk.loc[:, wanted_columns]

    # Drop rows that have NaN values in them
    df_uk = df_uk.dropna()

    # Create a DataSet for UK data that has values from both files
    df_uk = df_uk.merge(df_uk_added_data, on="FID")

    df_uk["sex_f31_0_0"].replace("Male", 1, inplace=True)
    df_uk["sex_f31_0_0"].replace("Female", 2, inplace=True)

    return df_uk


def __filter_il_data():
    # TODO: Maybe change to read straight from excel file
    df_il = pd.read_csv("../research/blood_test_il.csv")
    df_uk_added_data = pd.read_csv("../research/ICD10_UK.csv")

    # Filter dataSet values
    wanted_columns = list(values.features.keys())
    wanted_columns.extend(["hospital_patient_id"])

    df_il = df_il.loc[:, wanted_columns]

    # Keep only the latest test of the patient
    df_il = df_il.drop_duplicates(subset="hospital_patient_id", keep="last")

    # Drop rows that have NaN in them
    df_il = df_il.dropna()

    for illness in df_uk_added_data.columns[2:]:
        df_il[illness] = 3

    return df_il


def __merge_df(df_uk, df_il):

    # Change column names to match df_uk
    all_columns = {
        "hospital_patient_id": "FID",
    }
    all_columns.update(values.features)
    df_il = df_il.rename(columns=all_columns)

    df_uk = df_uk.drop(columns=["IID", "FID"])
    df_il = df_il.drop(columns=["FID"])

    # Merge datasets
    df_uk = df_uk.append(df_il)
    df_uk = df_uk.reset_index(drop=True)

    df_uk["D50*"] = df_uk["D500"]

    df_uk.loc[
        (df_uk["D500"] == 2)
        | (df_uk["D501"] == 2)
        | (df_uk["D508"] == 2)
        | (df_uk["D509"] == 2),
        "D50*",
    ] = 2

    df_uk = df_uk.drop(columns=["D500", "D501", "D508", "D509"])

    return df_uk


def create_data():
    """
        Creates the basic data for use
        Combines the UK dataset and the IL dataset into one
    """

    filtered_uk = __filter_uk_data()
    filtered_il = __filter_il_data()
    merged_df = __merge_df(df_uk=filtered_uk, df_il=filtered_il)

    return merged_df


def under_sample(data, fatty_liver=True):
    if fatty_liver:
        control = data[data["K760"] == 1]
        control = control.shape[0]
        print(control)
        ill = data[data["K760"] == 2]

        while ill.shape[0] < control:
            data.append(ill)
            data.reset_index()

            ill = data[data["K760"] == 2]

        print(ill.shape)


def mean_diff(data, pca=True):
    data = data.drop(columns=["K760", "D50*"])

    if pca:
        pca_data = PCA(n_components=len(values.features))
        data = pca_data.fit_transform(data)

    mean = data.mean(axis=0)

    return mean


def data_prep(data, test_size=0.3, random_state=13, fatty_liver=True):
    """ Splits the data into train and test, according to the test size sent. 
    Will return the results for Fatty Liver or Iron.
    Returns Background data for CPCA calculations.  """

    dataFrame = data[data["K760"] != 3]

    x = dataFrame.drop(columns=["K760", "D50*"])
    y1 = dataFrame["K760"]
    y2 = dataFrame["D50*"]

    df = data.loc[:, data.columns.difference(["K760", "D50*"])]

    background = df[(data["K760"] == 3) | (data["D50*"] == 3)]
    background = background.values

    if fatty_liver:
        X_train, X_test, y_train, y_test = train_test_split(
            x, y1, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test, background

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            x, y1, test_size=test_size, random_state=random_state
        )

        return X_train, X_test, y_train, y_test, background


def preform_cpca(X_train, X_test, background, alpha=1.06):
    """ Returns the Train and Test data after CPCA calculations. """
    mdl = CPCA(n_components=len(values.features))
    X_train = mdl.fit_transform(
        X_train, background, alpha_selection="manual", alpha_value=alpha
    )

    # Convert to NumPy array so CPCA calculation will work
    test = X_test.to_numpy()
    X_test = mdl.transform(test, alpha_selection="manual", alpha_value=alpha)

    return X_train, X_test


def preform_pca(X_train, X_test):
    """ Returns the Train and Test data after PCA calculations. """
    pca_data = PCA(n_components=len(values.features))
    X_train = pca_data.fit_transform(X_train)

    X_test = pca_data.transform(X_test)

    return X_train, X_test


def add_invalid_column(data, features_list):
    # If at least one test is not in the valid value range then value of column will be 0 (False)
    data["allTestValid"] = 1

    for key in features_list:
        # Gets data from dictonary
        min_value = features_list[key][0]
        max_value = features_list[key][1]

        # Change values of "allTestValid" according to test values in range
        data.loc[(data[key] > max_value) | (data[key] < min_value), "allTestValid"] = 0
        df = data.loc[(data[key] > max_value) | (data[key] < min_value)]
        print(key, df.shape)

    return data


def normalize_data(data):
    scaler = StandardScaler()
    transformed_data = scaler.fit_transform(data)

    return transformed_data


# Uses a different database to create the uk data
def create_data_new(path):
    df_uk = pd.read_csv(path)

    df_uk_added_data = pd.read_csv("../research/ICD10_ukbb_new.csv")

    df_uk_added_data_2 = pd.read_csv("../research/alzehimer_parkinson_asthma.csv")

    df_uk_added_data_2 = df_uk_added_data_2.drop(
        columns=["sex", "visit_date", "visit_age"]
    )

    print("All Data Shape: ", df_uk.shape)
    # Drop rows that have NaN values in them
    # Drop the Oestradiol (pmol/L) and Rheumatoid factor (IU/ml)
    # because the have too many NaN values

    # df_uk = df_uk.drop(columns=["Oestradiol (pmol/L)", "Rheumatoid factor (IU/ml)"])

    # df_uk = df_uk.dropna()
    # print("After dropping NaN rows: ", df_uk.shape)
    # Rename column for merge
    df_uk = df_uk.rename(columns={"eid": "FID"})
    df_uk_added_data_2 = df_uk_added_data_2.rename(columns={"eid": "FID"})

    # Create a DataSet for UK data that has values from both files
    df_uk = df_uk.merge(df_uk_added_data, on="FID", how="left")
    df_uk = df_uk.merge(df_uk_added_data_2, on="FID", how="left")

    df_uk["sex"].replace("Male", 1, inplace=True)
    df_uk["sex"].replace("Female", 2, inplace=True)

    df_uk["K760"] = df_uk["K760"] - 1

    # Create new columns for similar diseases
    df_uk["D50*"] = 0

    df_uk.loc[
        (df_uk["D500"] == 2)
        | (df_uk["D501"] == 2)
        | (df_uk["D508"] == 2)
        | (df_uk["D509"] == 2)
        | (df_uk["D630"] == 2)
        | (df_uk["D631"] == 2)
        | (df_uk["D638"] == 2)
        | (df_uk["D640"] == 2)
        | (df_uk["D641"] == 2)
        | (df_uk["D648"] == 2)
        | (df_uk["D642"] == 2)
        | (df_uk["D643"] == 2)
        | (df_uk["D644"] == 2),
        "D50*",
    ] = 1

    df_uk["D70*"] = 0

    df_uk.loc[
        (df_uk["D70"] == 2)
        | (df_uk["D700"] == 2)
        | (df_uk["D701"] == 2)
        | (df_uk["D702"] == 2)
        | (df_uk["D703"] == 2)
        | (df_uk["D704"] == 2)
        | (df_uk["D708"] == 2)
        | (df_uk["D709"] == 2),
        "D70*",
    ] = 1

    df_uk["alzheimer"] = 0
    df_uk.loc[(df_uk["Source of alzheimer's disease report"].notna(), "alzheimer")] = 1

    df_uk["parkinsonism"] = 0
    df_uk.loc[
        (df_uk["Source of all cause parkinsonism report"].notna(), "parkinsonism")
    ] = 1

    df_uk["asthma"] = 0
    df_uk.loc[(df_uk["Source of asthma report"].notna(), "asthma")] = 1

    df_uk = df_uk.drop(
        columns=[
            "D500",
            "D501",
            "D508",
            "D509",
            "D630",
            "D631",
            "D638",
            "D640",
            "D641",
            "D642",
            "D643",
            "D644",
            "D648",
            "D509",
            "D70",
            "D700",
            "D701",
            "D702",
            "D703",
            "D704",
            "D708",
            "D709",
            "visit_date",
            "Source of alzheimer's disease report",
            "Source of all cause parkinsonism report",
            "Source of asthma report",
        ]
    )
    print("After Dropping Phenotype Columns: ", df_uk.shape)

    return df_uk[df_uk["sex"] == 1], df_uk[df_uk["sex"] == 2]


def remove_columns_and_nan(data, features_list=False, illness=False):
    # Drop rows that have NaN values in them
    # Drop the Oestradiol (pmol/L) and Rheumatoid factor (IU/ml)
    # because the have too many NaN values

    # for x in features_list:
    #     print(x + " :", data[x].isnull().sum())

    if features_list == False:
        data = data.drop(columns=["Oestradiol (pmol/L)", "Rheumatoid factor (IU/ml)"])
    elif type(illness) is not list:
        keys_list = list(features_list.keys())
        keys_list.append(illness)
        data = data[keys_list]
    else:
        keys_list = list(features_list.keys())
        keys_list.extend(illness)
        data = data[keys_list]

    data = data.dropna()

    return data


def calc_num_of_illnesses(data):
    df = pd.read_csv("../research/new_illnesses.csv")

    cols_to_sum = df.columns[1:214]
    df["# of Illnesses"] = df[cols_to_sum].notna().sum(axis=1)

    # Drop all Columns except for eid and # of Illnesses
    df = df.filter(["eid", "# of Illnesses"], axis=1)
    df = df.rename(columns={"eid": "FID"})

    # Merge data
    data = data.merge(df, on="FID", how="left")

    return data

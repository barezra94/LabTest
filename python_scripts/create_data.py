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


def add_invalid_column(data):
    # If at least one test is not in the valid value range then value of column will be 0 (False)
    data["allTestValid"] = 1

    for key in values.data_values:
        # Gets data from dictonary
        name = values.data_values[key]["name"]
        min_value = values.data_values[key]["value"][0]
        max_value = values.data_values[key]["value"][1]

        # Change values of "allTestValid" according to test values in range
        data.loc[
            (data[name] > max_value) | (data[name] < min_value), "allTestValid"
        ] = 0

    return data


def normalize_data(data):
    scaler = StandardScaler()
    transformed_data = scaler.fit_transform(data)

    return transformed_data
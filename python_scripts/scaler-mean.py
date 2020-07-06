import create_data as cd
import scipy as sp

import numpy
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import values as vs

from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    data_male, data_female = cd.create_data_new("../research/ukbb_new_tests.csv")
    # data = cd.create_data()

    # Add invalid Test Column
    data_female = cd.add_invalid_column(data_female, vs.features_values_female)
    data_male = cd.add_invalid_column(data_male, vs.features_values_male)

    # Join to one dataframe for normalization
    data = pd.concat([data_female, data_male])

    # Normalize the Data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    # Get the mean of all the data rows
    mean = normalized_data.mean(axis=0)

    # Select all ill patients and all control patients, compute diff
    # and compute distance from overall mean
    ill_female = data[(data["D70*"] == 2) & (data["sex"] == 2)]
    ill_male = data[(data["D70*"] == 2) & (data["sex"] == 1)]
    control_female = data[(data["D70*"] == 1) & (data["sex"] == 2)]
    control_male = data[(data["D70*"] == 1) & (data["sex"] == 1)]

    print(ill_female.shape)
    print(ill_male.shape)

    # Normalize Ill and Control patients data
    normalized_ill_female = scaler.transform(ill_female)
    normalized_ill_male = scaler.transform(ill_male)
    normalized_control_female = scaler.transform(control_female)
    normalized_control_male = scaler.transform(control_male)

    mean = mean.reshape(1, -1)

    ill_diff_female = sp.spatial.distance.cdist(mean, normalized_ill_female)
    ill_diff_male = sp.spatial.distance.cdist(mean, normalized_ill_male)
    control_diff_female = sp.spatial.distance.cdist(mean, normalized_control_female)
    control_diff_male = sp.spatial.distance.cdist(mean, normalized_control_male)

    # Keep only rows that have at least one test that is not in range and compute diff
    invalid_range = data[data["allTestValid"] == 0]
    print(invalid_range.shape)
    normalized_invalid_range = scaler.transform(invalid_range)
    invalid_range_diff = sp.spatial.distance.cdist(mean, normalized_invalid_range)

    # Remove values that are high
    ill_diff_female = ill_diff_female[0]
    ill_diff_male = ill_diff_male[0]

    control_diff_female = control_diff_female[0]
    control_diff_male = control_diff_male[0]

    invalid_range_diff = invalid_range_diff[0]

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
    seaborn.distplot(
        control_diff_male,
        label="control-male",
        # hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )
    seaborn.distplot(
        ill_diff_male,
        label="ill-male",
        # hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )
    seaborn.distplot(
        invalid_range_diff,
        label="invalid range",
        # hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )

    plt.legend()
    plt.show()


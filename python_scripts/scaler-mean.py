import create_data as cd
import scipy as sp

import numpy
import seaborn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # data = cd.create_data_new("../research/ukbb_new_tests.csv")
    data = cd.create_data()

    # Add invalid Test Column
    data = cd.add_invalid_column(data)

    # Normalize the Data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    # Get the mean of all the data rows
    mean = normalized_data.mean(axis=0)

    # Select all ill patients and all control patients, compute diff
    # and compute distance from overall mean
    ill = data[data["K760"] == 2]
    control = data[data["K760"] == 1]

    # Normalize Ill and Control patients data
    normalized_ill = scaler.transform(ill)
    normalized_control = scaler.transform(control)
    print(normalized_data.shape)
    print(normalized_ill.shape)
    print(normalized_control.shape)

    mean = mean.reshape(1, -1)

    ill_diff = sp.spatial.distance.cdist(mean, normalized_ill)
    control_diff = sp.spatial.distance.cdist(mean, normalized_control)

    # Keep only rows that have at least one test that is not in range and compute diff
    invalid_range = data[data["allTestValid"] == 0]
    print(invalid_range.shape)
    normalized_invalid_range = scaler.transform(invalid_range)
    invalid_range_diff = sp.spatial.distance.cdist(mean, normalized_invalid_range)

    # Remove values that are high
    ill_diff = ill_diff[0]

    print(sorted(ill_diff)[:10])

    control_diff = control_diff[0]
    # while control_diff.argmax() > 120:
    # control_diff = numpy.delete(control_diff, control_diff.argmax())
    # control_diff = numpy.delete(control_diff, control_diff.argmax())
    # control_diff = numpy.delete(control_diff, control_diff.argmax())
    # control_diff = numpy.delete(control_diff, control_diff.argmax())
    # control_diff = numpy.delete(control_diff, control_diff.argmax())

    print(sorted(control_diff)[:10])

    invalid_range_diff = invalid_range_diff[0]

    print(sorted(invalid_range_diff)[:10])
    seaborn.distplot(
        control_diff,
        label="control",
        # hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )
    seaborn.distplot(
        ill_diff,
        label="ill",
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

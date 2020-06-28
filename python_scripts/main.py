import create_data as cd
import random_forest as rd
import logistic_regression as lg

import scipy as sp
import scipy.stats as st
import numpy
import seaborn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from contrastive import CPCA


def cpca(ill, control, dataframe, background):
    mdl = CPCA(n_components=len(cd.values.features))
    data_cpca = mdl.fit_transform(
        dataFrame, background, alpha_selection="manual", alpha_value=1.06
    )

    mean = data_cpca.mean(axis=0)

    ill_data = mdl.fit_transform(
        ill, background, alpha_selection="manual", alpha_value=1.06
    )
    control_data = mdl.fit_transform(
        control, background, alpha_selection="manual", alpha_value=1.06
    )

    mean = mean.reshape(1, -1)

    ill_diff = sp.spatial.distance.cdist(mean, ill_data)
    control_diff = sp.spatial.distance.cdist(mean, control_data)

    ill_diff = ill_diff.reshape(1, -1)
    control_diff = control_diff.reshape(1, -1)

    # Remove distances that are large
    ill_diff = ill_diff[0]

    control_diff = control_diff[0]
    control_diff = numpy.delete(control_diff, control_diff.argmax())

    ks_test2 = st.ks_2samp(control_diff, ill_diff)
    print(ks_test2)

    seaborn.distplot(
        control_diff,
        label="control",
        hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )

    seaborn.distplot(
        ill_diff,
        label="ill",
        hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )

    plt.legend()
    plt.show()

    print("Diff ill:", numpy.sort(ill_diff))
    print("Diff Control:", numpy.sort(control_diff))

    numpy.savetxt("1-cpca.csv", ill_diff, delimiter=",")
    numpy.savetxt("2-cpca.csv", control_diff, delimiter=",")


def pca(ill, control, mean):
    # Preform pca on the data
    pca_data = PCA(n_components=len(cd.values.features))
    data_after_pca = pca_data.fit_transform(data)

    ill = pca_data.fit_transform(ill)
    control = pca_data.fit_transform(control)
    ill = data_after_pca[ill.index]
    control = data_after_pca[control.index]

    mean = mean.reshape(1, -1)
    ill = ill.reshape(1, -1)
    control = control.reshape(1, -1)

    ill_diff = sp.spatial.distance.cdist(mean, ill)
    control_diff = sp.spatial.distance.cdist(mean, control)

    ill_diff = ill_diff.reshape(1, -1)
    control_diff = control_diff.reshape(1, -1)

    # Remove distances that are large
    ill_diff = ill_diff[0]

    control_diff = control_diff[0]
    control_diff = numpy.delete(control_diff, control_diff.argmax())

    seaborn.distplot(
        control_diff,
        label="control",
        hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )
    seaborn.distplot(
        ill_diff,
        label="ill",
        hist_kws={"cumulative": True},
        kde_kws={"cumulative": True},
    )

    plt.legend()
    plt.show()

    print("Diff ill:", ill_diff)
    print("Diff Control:", control_diff)

    numpy.savetxt("1-pca.csv", ill_diff, delimiter=",")
    numpy.savetxt("2-pca.csv", control_diff, delimiter=",")


if __name__ == "__main__":
    data = cd.create_data()
    mean = cd.mean_diff(data)

    # Select an ill patient and a control patient and check the diff from mean
    ill = data[data["K760"] == 2]
    control = data[data["K760"] == 1]

    dataFrame = data[data["K760"] != 3]

    background = data[(data["K760"] == 3) | (data["D50*"] == 3)]
    background = background.values

    cpca(ill, control, dataFrame, background)
    # pca(ill, control, mean)

    # K-S Test
    # ill_control = numpy.concatenate((control_diff, ill_diff))
    # ks_test2 = st.ks_2samp(control_diff, ill_diff)

    # ill_cdf = [
    #     numpy.round(st.percentileofscore(ill, samp) / 100, 1) for samp in ill_control
    # ]

    # control_cdf = [
    #     numpy.round(st.percentileofscore(control, samp) / 100, 1)
    #     for samp in ill_control
    # ]

    # plt.plot(ill_control, ill_cdf, label="ill", alpha=0.5, marker="o", color="red")
    # plt.plot(
    #     ill_control, control_cdf, label="control", alpha=0.5, marker="o", color="blue"
    # )

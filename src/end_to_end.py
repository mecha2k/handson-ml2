import os
import sys
import warnings

import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"


def main():
    mpl.rc("axes", labelsize=14)
    mpl.rc("xtick", labelsize=12)
    mpl.rc("ytick", labelsize=12)

    warnings.filterwarnings(action="ignore", message="^internal gelsd")

    housing_path = "../master/datasets/housing"
    csv_path = os.path.join(housing_path, "housing.csv")

    housing = pd.read_csv(csv_path)
    print(housing.head())

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=100)
    print("train set: ", len(train_set))

    housing["income_cat"] = pd.cut(
        housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]
    )

    global strat_train_set, strat_test_set
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    print(housing["income_cat"].value_counts() / len(housing))
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

    images_path = "../master/images/end_to_end_project/california.png"
    california_img = mpimg.imread(images_path)
    ax = housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        figsize=(10, 7),
        s=housing["population"] / 100,
        label="Population",
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=False,
        alpha=0.4,
    )
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5, cmap=plt.get_cmap("jet"))
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)

    prices = housing["median_house_value"]
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    cbar = plt.colorbar(ticks=tick_values / prices.max())
    cbar.ax.set_yticklabels(["$%dk" % (round(v / 1000)) for v in tick_values], fontsize=14)
    cbar.set_label("Median House Value", fontsize=16)

    plt.legend(fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()

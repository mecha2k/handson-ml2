import os
import sys
import warnings

import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

assert sys.version_info >= (3, 5)
assert sklearn.__version__ >= "0.20"


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # column index
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


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

    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()

    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    print(sample_incomplete_rows)

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    print(imputer.statistics_)
    print(housing_num.median().values)

    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    print(housing_tr.loc[sample_incomplete_rows.index.values])
    print(housing_tr.isnull().sum())
    print(housing_tr.info())

    housing_cat = housing[["ocean_proximity"]]

    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print(ordinal_encoder.categories_)
    print(housing_cat_encoded[:10])

    cat_encoder = OneHotEncoder(sparse=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print(housing_cat_1hot)

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)
    print(housing.index)
    print(housing.columns)
    print(housing.values)

    housing_extra_attribs = pd.DataFrame(
        housing_extra_attribs,
        columns=list(housing.columns) + ["rooms_per_household", "population_per_household"],
        index=housing.index,
    )
    print(housing_extra_attribs.info())

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ]
    )
    housing_prepared = full_pipeline.fit_transform(housing)

    def display_scores(scores):
        print("Scores:", scores)
        print("Mean:", scores.mean())
        print("Standard deviation:", scores.std())

    # lin_reg = LinearRegression()
    # lin_reg.fit(housing_prepared, housing_labels)
    # housing_predictions = lin_reg.predict(housing_prepared)
    # lin_mse = mean_squared_error(housing_labels, housing_predictions)
    # lin_rmse = np.sqrt(lin_mse)
    # lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    # print("linear rmse, mae:", lin_rmse, lin_mae)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    display_scores(tree_rmse_scores)

    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_reg.fit(housing_prepared, housing_labels)
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    forest_scores = cross_val_score(
        forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10
    )
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)


if __name__ == "__main__":
    main()

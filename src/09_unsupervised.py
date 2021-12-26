import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import urllib

from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from matplotlib.colors import LogNorm
from sklearn.metrics import silhouette_score
from matplotlib.image import imread
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN


def plot_data(X_):
    plt.plot(X_[:, 0], X_[:, 1], "k.", markersize=2)


def plot_decision_boundaries(cluster, X_, resolution_=1000, show_centroids=True, show_xlabels=True, show_ylabels=True):
    mins = X_.min(axis=0) - 0.1
    maxs = X_.max(axis=0) + 0.1
    xx_, yy_ = np.meshgrid(np.linspace(mins[0], maxs[0], resolution_), np.linspace(mins[1], maxs[1], resolution_))
    Z = cluster.predict(np.c_[xx_.ravel(), yy_.ravel()])
    Z = Z.reshape(xx_.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidths=1, colors="k")
    plot_data(X_)
    if show_centroids:
        plot_centroids(cluster.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


def plot_clusterer_comparison(clusterer1, clusterer2, X_, title1=None, title2=None):
    clusterer1.fit(X_)
    clusterer2.fit(X_)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X_)
    if title1:
        plt.title(title1, fontsize=14)

    plt.subplot(122)
    plot_decision_boundaries(clusterer2, X_, show_ylabels=False)
    if title2:
        plt.title(title2, fontsize=14)


def plot_dbscan(dbscan_, X_, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan_.labels_, dtype=bool)
    core_mask[dbscan_.core_sample_indices_] = True
    anomalies_mask = dbscan_.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan_.components_
    anomalies = X_[anomalies_mask]
    non_cores = X_[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1], c=dbscan_.labels_[core_mask], marker="o", s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker="*", s=20, c=dbscan_.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan_.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("eps={:.2f}, min_samples={}".format(dbscan_.eps, dbscan_.min_samples), fontsize=14)


def plot_centroids(centroids, weights=None, circle_color="w", cross_color="k"):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="o",
        s=30,
        linewidths=8,
        color=circle_color,
        zorder=10,
        alpha=0.9,
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=None,
        linewidths=None,
        color=cross_color,
        zorder=11,
        alpha=1,
    )


def plot_gaussian_mixture(clusterer, X_, resolution_=1000, show_ylabels=True):
    mins = X_.min(axis=0) - 0.1
    maxs = X_.max(axis=0) + 0.1
    xx_, yy_ = np.meshgrid(np.linspace(mins[0], maxs[0], resolution_), np.linspace(mins[1], maxs[1], resolution_))
    Z = -clusterer.score_samples(np.c_[xx_.ravel(), yy_.ravel()])
    Z = Z.reshape(xx_.shape)

    plt.contourf(xx_, yy_, Z, norm=LogNorm(vmin=1.0, vmax=30.0), levels=np.logspace(0, 2, 12))
    plt.contour(
        xx_,
        yy_,
        Z,
        norm=LogNorm(vmin=1.0, vmax=30.0),
        levels=np.logspace(0, 2, 12),
        linewidths=1,
        colors="k",
    )

    Z = clusterer.predict(np.c_[xx_.ravel(), yy_.ravel()])
    Z = Z.reshape(xx_.shape)
    plt.contour(xx_, yy_, Z, linewidths=2, colors="r", linestyles="dashed")

    plt.plot(X_[:, 0], X_[:, 1], "k.", markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


def compare_gaussian_mixtures(gm1, gm2, X):
    plt.figure(figsize=(9, 4))

    plt.subplot(121)
    plot_gaussian_mixture(gm1, X)
    plt.title('covariance_type="{}"'.format(gm1.covariance_type), fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(gm2, X, show_ylabels=False)
    plt.title('covariance_type="{}"'.format(gm2.covariance_type), fontsize=14)


def runGaussianMixture():
    X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    X2 = X2 + [6, -8]
    X = np.r_[X1, X2]

    gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
    gm.fit(X)

    print(gm.means_)
    print(gm.weights_)
    print(gm.covariances_)
    print(gm.converged_)
    print(gm.n_iter_)
    print(gm.predict(X))
    score = gm.score_samples(X)
    print(score)

    x_new, y_new = gm.sample(10)
    print(x_new)
    print(y_new)

    resolution = 100
    grid = np.arange(-10, 10, 1 / resolution)
    xx, yy = np.meshgrid(grid, grid)
    X_full = np.vstack([xx.ravel(), yy.ravel()]).T

    pdf = np.exp(gm.score_samples(X_full))
    pdf_probas = pdf * (1 / resolution) ** 2
    print(pdf_probas.sum())

    plt.figure(figsize=(8, 4))
    plot_gaussian_mixture(gm, X)
    plt.tight_layout()
    plt.savefig("images/ch08/08_gaussian_mixtures.png", format="png", dpi=300)

    gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
    gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
    gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
    gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)
    gm_full.fit(X)
    gm_tied.fit(X)
    gm_spherical.fit(X)
    gm_diag.fit(X)

    compare_gaussian_mixtures(gm_tied, gm_spherical, X)
    plt.tight_layout()
    plt.savefig("images/ch08/08_covariance_type_plot_01.png", format="png", dpi=300)

    compare_gaussian_mixtures(gm_full, gm_diag, X)
    plt.tight_layout()
    plt.savefig("images/ch08/08_covariance_type_plot_02.png", format="png", dpi=300)

    densities = gm.score_samples(X)
    density_threshold = np.percentile(densities, 4)
    anomalies = X[densities < density_threshold]

    plt.figure(figsize=(8, 4))
    plot_gaussian_mixture(gm, X)
    plt.scatter(anomalies[:, 0], anomalies[:, 1], color="r", marker="*")
    plt.ylim(top=5.1)
    plt.savefig("images/ch08/08_mixture_anomaly_detection_plot.png", format="png", dpi=300)

    gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X) for k in range(1, 11)]

    bics = [model.bic(X) for model in gms_per_k]
    aics = [model.aic(X) for model in gms_per_k]

    plt.figure(figsize=(8, 3))
    plt.plot(range(1, 11), bics, "bo-", label="BIC")
    plt.plot(range(1, 11), aics, "go--", label="AIC")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Information Criterion", fontsize=14)
    plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])
    plt.annotate(
        "Minimum",
        xy=(3, bics[2]),
        xytext=(0.35, 0.6),
        textcoords="figure fraction",
        fontsize=14,
        arrowprops=dict(facecolor="black", shrink=0.1),
    )
    plt.legend()
    plt.savefig("images/ch08/08_aic_bic_vs_k_plot.png", format="png", dpi=300)

    min_bic = np.infty
    best_k, best_covariance_type = None, None
    for k in range(1, 11):
        for covariance_type in ("full", "tied", "spherical", "diag"):
            bic = (
                GaussianMixture(n_components=k, n_init=10, covariance_type=covariance_type, random_state=42)
                .fit(X)
                .bic(X)
            )
            if bic < min_bic:
                min_bic = bic
                best_k = k
                best_covariance_type = covariance_type

    print(best_k)
    print(best_covariance_type)


def runVariationalBayesianGaussian():
    X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    X2 = X2 + [6, -8]
    X = np.r_[X1, X2]

    bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
    bgm.fit(X)
    np.round(bgm.weights_, 2)

    plt.figure(figsize=(8, 5))
    plot_gaussian_mixture(bgm, X)
    plt.savefig("images/ch08/08_gaussian_mixture_01.png", dpi=300, bboxinches="tight")
    
    bgm_low = BayesianGaussianMixture(
        n_components=10, max_iter=1000, n_init=1, weight_concentration_prior=0.01, random_state=42
    )
    bgm_high = BayesianGaussianMixture(
        n_components=10, max_iter=1000, n_init=1, weight_concentration_prior=10000, random_state=42
    )
    nn = 73
    bgm_low.fit(X[:nn])
    bgm_high.fit(X[:nn])

    np.round(bgm_low.weights_, 2)
    np.round(bgm_high.weights_, 2)

    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plot_gaussian_mixture(bgm_low, X[:nn])
    plt.title("weight_concentration_prior = 0.01", fontsize=14)
    plt.subplot(122)
    plot_gaussian_mixture(bgm_high, X[:nn], show_ylabels=False)
    plt.title("weight_concentration_prior = 10000", fontsize=14)
    plt.savefig("images/ch08/08_gaussian_mixture_02.png", dpi=300, bboxinches="tight")


def runGaussianMixture_iris():
    data = load_iris()
    X = data.data
    y = data.target
    print(data.DESCR)
    print(data.target_names)

    y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)
    mapping = np.array([2, 0, 1])
    y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

    print(X[0], X[0, 2], X[0][2])
    print(X[y_pred == 0, 2])
    print(f"accuracy : {np.sum(y_pred==y) / len(y_pred) * 100.0:.1f}%")

    plt.figure(figsize=(8, 4))
    plt.plot(X[y_pred == 0, 2], X[y_pred == 0, 3], "yo", label="Cluster 1")
    plt.plot(X[y_pred == 1, 2], X[y_pred == 1, 3], "bs", label="Cluster 2")
    plt.plot(X[y_pred == 2, 2], X[y_pred == 2, 3], "g^", label="Cluster 3")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.savefig("images/ch08/08_gaussian_mixture_iris.png", dpi=300, bboxinches="tight")


def runKMeans_blobs():
    blob_centers = np.array([[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]])
    blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)

    plt.figure(figsize=(8, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.savefig("images/ch08/08_kmeans_blob.png", dpi=300, bboxinches="tight")

    kmeans = KMeans(n_clusters=5, random_state=42)
    y_pred = kmeans.fit_predict(X)
    print(kmeans.cluster_centers_)
    print(kmeans.labels_)

    plt.figure(figsize=(8, 4))
    plot_decision_boundaries(kmeans, X)
    plt.savefig("images/ch08/08_voronoi.png", dpi=300, bboxinches="tight")

    X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
    print(kmeans.predict(X_new))
    print(kmeans.transform(X_new))
    print(np.linalg.norm(np.tile(X_new, (1, 5)).reshape(-1, 5, 2) - kmeans.cluster_centers_, axis=2))

    kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", max_iter=1, random_state=1)
    kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", max_iter=2, random_state=1)
    kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", max_iter=3, random_state=1)
    kmeans_iter1.fit(X)
    kmeans_iter2.fit(X)
    kmeans_iter3.fit(X)

    plt.figure(figsize=(10, 8))
    plt.subplot(321)
    plot_data(X)
    plot_centroids(kmeans_iter1.cluster_centers_, circle_color="r", cross_color="w")
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.tick_params(labelbottom=False)
    plt.title("Update the centroids (initially randomly)", fontsize=14)
    plt.subplot(322)
    plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
    plt.title("Label the instances", fontsize=14)
    plt.subplot(323)
    plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
    plot_centroids(kmeans_iter2.cluster_centers_)
    plt.subplot(324)
    plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)
    plt.subplot(325)
    plot_decision_boundaries(kmeans_iter2, X, show_centroids=False)
    plot_centroids(kmeans_iter3.cluster_centers_)
    plt.subplot(326)
    plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)
    plt.savefig("images/ch08/08_kmeans_algorithm.png", dpi=300, bboxinches="tight")

    kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", random_state=11)
    kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1, algorithm="full", random_state=19)

    plot_clusterer_comparison(
        kmeans_rnd_init1, kmeans_rnd_init2, X, "Solution 1", "Solution 2 (with a different random init)"
    )
    plt.savefig("images/ch08/08_kmeans_cluster_comparison.png", dpi=300, bboxinches="tight")

    minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
    minibatch_kmeans.fit(X)

    kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X) for k in range(1, 10)]
    inertias = [model.inertia_ for model in kmeans_per_k]
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 10), inertias, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.annotate(
        "Elbow",
        xy=(4, inertias[3]),
        xytext=(0.55, 0.55),
        textcoords="figure fraction",
        fontsize=16,
        arrowprops=dict(facecolor="black", shrink=0.1),
    )
    plt.axis([1, 8.5, 0, 1300])
    plt.savefig("images/ch08/08_kmeans_sse_diagram.png", dpi=300, bboxinches="tight")

    silhouette_scores = [silhouette_score(X, model.labels_) for model in kmeans_per_k[1:]]

    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 10), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.axis([1.8, 8.5, 0.55, 0.7])
    plt.savefig("images/ch08/08_kmeans_silhouette.png", dpi=300, bboxinches="tight")

    plt.figure(figsize=(11, 9))
    for k in (3, 4, 5, 6):
        plt.subplot(2, 2, k - 2)
        y_pred = kmeans_per_k[k - 1].labels_
        silhouette_coefficients = silhouette_samples(X, y_pred)
        padding = len(X) // 30
        pos = padding
        ticks = []
        for i in range(k):
            coeffs = silhouette_coefficients[y_pred == i]
            coeffs.sort()
            color = matplotlib.cm.Spectral(i / k)
            plt.fill_betweenx(
                np.arange(pos, pos + len(coeffs)),
                0,
                coeffs,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
            ticks.append(pos + len(coeffs) // 2)
            pos += len(coeffs) + padding

        plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
        plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
        if k in (3, 5):
            plt.ylabel("Cluster")
        if k in (5, 6):
            plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.xlabel("Silhouette Coefficient")
        else:
            plt.tick_params(labelbottom=False)
        plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
        plt.title("$k={}$".format(k), fontsize=16)
    plt.savefig("images/ch08/08_kmeans_silhouette_coeff.png", dpi=300, bboxinches="tight")


def runImgSeg():
    os.makedirs("images/ch08/unsupervised", exist_ok=True)
    url = "https://github.com/ageron/handson-ml2/blob/master/images/unsupervised_learning/ladybug.png"
    # urllib.request.urlretrieve(url, "images/unsupervised/ladybug.png")

    image = imread("images/ch08/unsupervised/ladybug.png")
    print(image.shape)

    X = image.reshape(-1, 3)
    segmented_imgs = []
    n_colors = (10, 8, 6, 4, 2)
    for n_clusters in n_colors:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        segmented_imgs.append(segmented_img.reshape(image.shape))

    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.subplot(231)
    plt.imshow(image)
    plt.title("Original image")
    plt.axis("off")
    for idx, n_clusters in enumerate(n_colors):
        plt.subplot(232 + idx)
        plt.imshow(segmented_imgs[idx])
        plt.title("{} colors".format(n_clusters))
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("images/ch08/08_image_seg_diagram.png", dpi=300, bboxinches="tight")


def runKMeans_digits():
    X_digits, y_digits = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    log_reg.fit(X_train, y_train)
    print(log_reg.score(X_test, y_test))

    pipeline = Pipeline(
        [
            ("kmeans", KMeans(n_clusters=50, random_state=42)),
            (
                "log_reg",
                LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    print(pipeline.score(X_test, y_test))

    param_grid = dict(kmeans__n_clusters=range(99, 100))
    grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
    grid_clf.fit(X_train, y_train)
    print(grid_clf.best_params_)
    print(grid_clf.score(X_test, y_test))

    n_labeled = 50
    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
    log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
    print(log_reg.score(X_test, y_test))

    k = 50
    kmeans = KMeans(n_clusters=k, random_state=42)
    X_digits_dist = kmeans.fit_transform(X_train)
    representative_digit_idx = np.argmin(X_digits_dist, axis=0)
    X_representative_digits = X_train[representative_digit_idx]

    plt.figure(figsize=(8, 2))
    for index, X_representative_digit in enumerate(X_representative_digits):
        plt.subplot(k // 10, 10, index + 1)
        plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
        plt.axis("off")
    plt.savefig("images/ch08/08_KMeans_digits.png", dpi=300, bboxinches="tight")

    # fmt: off
    y_representative_digits = np.array([
        4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
        5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
        1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
        6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
        4, 2, 9, 4, 7, 6, 2, 3, 1, 1])
    # fmt: on
    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    log_reg.fit(X_representative_digits, y_representative_digits)
    print(log_reg.score(X_test, y_test))

    y_train_propagated = np.empty(len(X_train), dtype=np.int32)
    for i in range(k):
        y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]
    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    log_reg.fit(X_train, y_train_propagated)
    print(log_reg.score(X_test, y_test))

    percentile_closest = 20

    X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
    for i in range(k):
        in_cluster = kmeans.labels_ == i
        cluster_dist = X_cluster_dist[in_cluster]
        cutoff_distance = np.percentile(cluster_dist, percentile_closest)
        above_cutoff = X_cluster_dist > cutoff_distance
        X_cluster_dist[in_cluster & above_cutoff] = -1

    partially_propagated = X_cluster_dist != -1
    X_train_partially_propagated = X_train[partially_propagated]
    y_train_partially_propagated = y_train_propagated[partially_propagated]

    log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
    log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
    print(log_reg.score(X_test, y_test))
    print(np.mean(y_train_partially_propagated == y_train[partially_propagated]))


def runDBSCAN():
    X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

    dbscan = DBSCAN(eps=0.05, min_samples=5)
    dbscan.fit(X)
    dbscan2 = DBSCAN(eps=0.2)
    dbscan2.fit(X)
    print(np.unique(dbscan.labels_))

    plt.figure(figsize=(9, 4))
    plt.subplot(121)
    plot_dbscan(dbscan, X, size=100)
    plt.subplot(122)
    plot_dbscan(dbscan2, X, size=600, show_ylabels=False)
    plt.savefig("images/ch08/08_dbscan.png", dpi=300, bboxinches="tight")

    dbscan = dbscan2
    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

    X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
    print(knn.predict(X_new))
    print(knn.predict_proba(X_new))

    plt.figure(figsize=(6, 3))
    plot_decision_boundaries(knn, X, show_centroids=False)
    plt.scatter(X_new[:, 0], X_new[:, 1], c="b", marker="+", s=200, zorder=10)
    plt.savefig("images/ch08/08_dbscan_knn.png", dpi=300, bboxinches="tight")


if __name__ == "__main__":
    runKMeans_blobs()
    runGaussianMixture_iris()
    # runImgSeg()
    # runKMeans_digits()
    # runDBSCAN()
    # runGaussianMixture()
    # runVariationalBayesianGaussian()

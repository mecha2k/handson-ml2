"""
Gaussian Mixture Model
Book : Mathematics for Machine Learning - chap 11
"""

import numpy as np
import matplotlib.pyplot as plt


def pdf(data, mean: float, variance: float):
    # A normal continuous random variable.
    s1 = 1 / (np.sqrt(2 * np.pi * variance))
    s2 = np.exp(-(np.square(data - mean) / (2 * variance)))
    return s1 * s2


X = np.array([-3, -2.5, -1, 0, 2, 4, 5]).astype(float)
means = np.array([-4, 0, 8]).astype(float)
variances = np.array([1, 0.2, 3]).astype(float)
weights = np.array([1 / 3, 1 / 3, 1 / 3]).astype(float)


eps = 1e-8
k = means.shape[0]
bins = np.linspace(np.min(X), np.max(X), 100)

for step in range(5):
    if step % 1 == 0:
        plt.figure(figsize=(10, 6))
        axes = plt.gca()
        plt.xlabel("$x$")
        plt.ylabel("pdf")
        plt.title("Iteration {}".format(step))
        plt.scatter(X, [0.005] * len(X), color="navy", s=30, marker=2, label="Train data")

        plt.plot(bins, pdf(bins, means[0], variances[0]), color="blue", label="Cluster 1")
        plt.plot(bins, pdf(bins, means[1], variances[1]), color="green", label="Cluster 2")
        plt.plot(bins, pdf(bins, means[2], variances[2]), color="magenta", label="Cluster 3")
        plt.legend(loc="upper left")
        plt.show()

    # calculate the maximum likelihood of each observation xi
    likelihood = []
    for j in range(k):
        likelihood.append(pdf(X, means[j], variances[j]))
    likelihood = np.array(likelihood)

    # Maximization step
    b = []
    for j in range(k):
        # use the current values for the parameters to evaluate the posterior
        # probabilities of the data to have been generanted by each gaussian
        likelihood_sum = np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0) + eps
        b.append((likelihood[j] * weights[j]) / likelihood_sum)

        # updage mean and variance
        means[j] = np.sum(b[j] * X) / (np.sum(b[j] + eps))
        variances[j] = np.sum(b[j] * np.square(X - means[j])) / (np.sum(b[j] + eps))

        # update the weights
        weights[j] = np.mean(b[j])

    print(means)
    print(variances)
    print(weights)


# for nloop in range(10):
#     mu_old = mu.sum()
#
#     rnk = np.zeros((n, k))
#     for i in range(n):
#         rnk_sum = 0
#         for j in range(k):
#             rnk_sum += pi[j] * pdf(Xn[i], mu[j], sg[j])
#         for j in range(k):
#             rnk[i][j] = pi[j] * pdf(Xn[i], mu[j], sg[j]) / (rnk_sum + eps)
#
#     N_k = np.zeros(k)
#     for i in range(k):
#         mu[i] = 0
#         for j in range(n):
#             N_k[i] += rnk[j][i]
#             mu[i] += rnk[j][i] * Xn[j]
#         mu[i] = mu[i] / (N_k[i] + eps)
#
#     for i in range(k):
#         sg[i] = 0
#         for j in range(n):
#             sg[i] += rnk[j][i] * (Xn[j] - mu[i]) * (Xn[j] - mu[i])
#         sg[i] = sg[i] / (N_k[i] + eps)
#         pi[i] = N_k[i] / float(n)
#
#     diff = abs(mu_old - mu.sum())
#
#     if diff < 1e-3:
#         break
#     print(nloop, diff, mu)
#
# print(mu)
# print(sg)
# print(pi)


# import matplotlib.pyplot as plt
# from sklearn import cluster, datasets, mixture
# import numpy as np
# from scipy.stats import multivariate_normal
#
# n_samples = 100
# mu1, sigma1 = -4, 1.2  # mean and variance
# mu2, sigma2 = 4, 1.8  # mean and variance
# mu3, sigma3 = 0, 1.6  # mean and variance
#
# x1 = np.random.normal(mu1, np.sqrt(sigma1), n_samples)
# x2 = np.random.normal(mu2, np.sqrt(sigma2), n_samples)
# x3 = np.random.normal(mu3, np.sqrt(sigma3), n_samples)
#
# X = np.array(list(x1) + list(x2) + list(x3))
# np.random.shuffle(X)
# print("Dataset shape:", X.shape)
#
#
# def pdf(data, mean: float, variance: float):
#     # A normal continuous random variable.
#     s1 = 1 / (np.sqrt(2 * np.pi * variance))
#     s2 = np.exp(-(np.square(data - mean) / (2 * variance)))
#     return s1 * s2
#
#
# # visualize the training data
# bins = np.linspace(np.min(X), np.max(X), 100)
#
# plt.figure(figsize=(8, 5))
# plt.xlabel("$x$", fontsize=14)
# plt.ylabel("pdf", fontsize=14)
# plt.scatter(X, [0.005] * len(X), color="navy", s=30, marker=2, label="Train data")
#
# plt.plot(bins, pdf(bins, mu1, sigma1), color="red", label="True pdf")
# plt.plot(bins, pdf(bins, mu2, sigma2), color="red")
# plt.plot(bins, pdf(bins, mu3, sigma3), color="red")
#
# plt.legend()
# plt.plot()
# plt.show()
#
# define the number of clusters to be learned
# k = 3
# weights = np.ones(k) / k
# means = np.random.choice(X, k)
# variances = np.random.random_sample(size=k)
#
# print(means, variances)
#
# X = np.array(X)
# print(X.shape)
#
# eps = 1e-8
# for step in range(10):
#
#     if step % 5 == 0:
#         plt.figure(figsize=(10, 6))
#         axes = plt.gca()
#         plt.xlabel("$x$")
#         plt.ylabel("pdf")
#         plt.title("Iteration {}".format(step))
#         plt.scatter(X, [0.005] * len(X), color="navy", s=30, marker=2, label="Train data")
#
#         plt.plot(bins, pdf(bins, mu1, sigma1), color="grey", label="True pdf")
#         plt.plot(bins, pdf(bins, mu2, sigma2), color="grey")
#         plt.plot(bins, pdf(bins, mu3, sigma3), color="grey")
#
#         plt.plot(bins, pdf(bins, means[0], variances[0]), color="blue", label="Cluster 1")
#         plt.plot(bins, pdf(bins, means[1], variances[1]), color="green", label="Cluster 2")
#         plt.plot(bins, pdf(bins, means[2], variances[2]), color="magenta", label="Cluster 3")
#
#         plt.legend(loc="upper left")
#
#         # plt.savefig("img_{0:02d}".format(step), bbox_inches="tight")
#         plt.show()
#
#     # calculate the maximum likelihood of each observation xi
#     likelihood = []
#
#     # Expectation step
#     for j in range(k):
#         likelihood.append(pdf(X, means[j], np.sqrt(variances[j])))
#     likelihood = np.array(likelihood)
#
#     b = []
#     # Maximization step
#     for j in range(k):
#         # use the current values for the parameters to evaluate the posterior
#         # probabilities of the data to have been generanted by each gaussian
#         b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0) + eps))
#
#         # updage mean and variance
#         means[j] = np.sum(b[j] * X) / (np.sum(b[j] + eps))
#         variances[j] = np.sum(b[j] * np.square(X - means[j])) / (np.sum(b[j] + eps))
#
#         # update the weights
#         weights[j] = np.mean(b[j])
#
#
# print(means)
# print(variances)
# print(weights)

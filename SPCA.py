__author__ = 'osopova'

from math import sqrt

from sklearn.utils import array2d, as_float_array, atleast2d_or_csr
from sklearn.utils.extmath import safe_sparse_dot

from scipy import linalg

import numpy as np

class SPCA():

    def __init__(self, components):
        # number of components returned with maximum variance
        self.n_components = components

        # array of components with maximum variance
        self.components_ = None
    #--------------------------------------------------------------------------------------------------------------

    def transform(self, features):

        features = atleast2d_or_csr(features)

        if self.mean_ is not None:
            features = features - self.mean_

        features = safe_sparse_dot(features, self.components_.T)
        return features
    #--------------------------------------------------------------------------------------------------------------

    def fit(self, features_train):

        X = features_train

        X = array2d(X)
        n_samples, n_features = X.shape

        X = as_float_array(X)
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, V = linalg.svd(X, full_matrices=False)
        explained_variance_ = (S ** 2) / n_samples
        explained_variance_ratio_ = (explained_variance_ /
                                     explained_variance_.sum())

        components_ = V / (S[:, np.newaxis] / sqrt(n_samples))

        n_components = self.n_components

        if n_components is None:
            n_components = n_features
        elif not 0 <= n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d" % (n_components, n_features))

        if 0 < n_components < 1.0:
            # number of components for which the cumulated explained variance
            # percentage is superior to the desired threshold
            ratio_cumsum = explained_variance_ratio_.cumsum()
            n_components = np.sum(ratio_cumsum < n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < n_features:
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        # store n_samples to revert whitening when getting covariance
        self.n_samples_ = n_samples

        self.components_ = components_[:n_components]
        #self.explained_variance_ = explained_variance_[:n_components]
        #explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        #self.explained_variance_ratio_ = explained_variance_ratio_
        self.n_components_ = n_components

        return (U, S, V)
    #--------------------------------------------------------------------------------------------------------------


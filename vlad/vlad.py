"""
DOCSTRING
"""
from random import sample

import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import progressbar as pb


class VLAD:
    """VLAD - Vector of Locally Aggregated Descriptors

    This class provides an implementation of the original proposal of "Vector of Locally Aggregated Descriptors" (VLAD)
    originally proposed in [1]_.

    Parameters
    ----------
    k : int, default=256
        Number of clusters to obtain for the visual vocabulary.
    n_vocabs : int, default=1
        Number of vocabularies to use
    norming : {"original", "intra", "RN"}, default="original"
        How the norming of the VLAD-descriptors should be performed.
        For more info see below.
    lcs : bool, default=True
        If `True`, uses Local Coordinate System (LCS) described in [3]_.
    alpha : float, default=0.2
        The exponent for the root-part, default taken from [3]_
    aggregate : str, {'bagging', 'concat'}, default='bagging'
        How to aggregate the VLAD-vectors when using multiple vocabularies
    verbose : bool, default=True
        If `True` print messages here and there

    Attributes
    ----------
    vocabs : sklearn.cluster.KMeans(k)
        The visual vocabulary of the object
    centers : array
        The centroids for the visual vocabulary
    database : array
        All known VLAD-vectors

    Notes
    -----
    ``norming="original"`` uses the original formulation of [1]_. An updated formulation based on [2]_
    is provided by ``norming="intra"``. Finally the best norming based on [3]_ is provided by ``norming="RN"``.

    The ``aggregate``-parameter either concatenates the individual VLAD-vectors followed by $L_2$-normalization
    when ``aggregate='concat'`` or performs mean-pooling between the `n_vocabs` VLAD-vectors and thus performs
    bagging. The effect is expected to be the same as sklearns ``VotingClassifier`` with ``voting`` set to ``soft``.
    The mean-pooling is performed to keep scores comparable. The effect of ``bagging='concat'`` is the behavior found
    in the literature, for example in [4]_ or [5]_.

    References
    ----------
    .. [1] Jégou, H., Douze, M., Schmid, C., & Pérez, P. (2010, June). Aggregating
           local descriptors into a compact image representation. In 2010 IEEE computer
           society conference on computer vision and pattern recognition (pp. 3304-3311). IEEE.

    .. [2] Arandjelovic, R., & Zisserman, A. (2013). All about VLAD. In Proceedings of the
           IEEE conference on Computer Vision and Pattern Recognition (pp. 1578-1585).

    .. [3] Delhumeau, J., Gosselin, P. H., Jégou, H., & Pérez, P. (2013, October).
           Revisiting the VLAD image representation. In Proceedings of the 21st ACM
           international conference on Multimedia (pp. 653-656).

    .. [4] Jégou, H., & Chum, O. (2012, October). Negative evidences and co-occurences in image
           retrieval: The benefit of PCA and whitening. In European conference on computer vision
           (pp. 774-787). Springer, Berlin, Heidelberg.

    .. [5] Spyromitros-Xioufis, E., Papadopoulos, S., Kompatsiaris, I. Y., Tsoumakas, G., & Vlahavas,
           I. (2014). A comprehensive study over VLAD and product quantization in large-scale image retrieval.
           IEEE Transactions on Multimedia, 16(6), 1713-1728.
    """
    def __init__(self, k=256, n_vocabs=1, norming="original", lcs=False, alpha=0.2, aggregate='bagging', verbose=True):
        self.k = k
        self.n_vocabs = n_vocabs
        self.norming = norming
        self.vocabs = None
        self.centers = None
        self.database = None
        self.lcs = lcs
        self.alpha = alpha
        self.qs = None
        self.aggregate = aggregate
        self.verbose = verbose

    def fit(self, X):
        """Fit Visual Vocabulary

        Parameters
        ----------
        X : list(array)
            List of image descriptors

        Returns
        -------
        self : VLAD
            Fitted object
        """
        X_mat = np.vstack(X)
        self.vocabs = []
        self.centers = []
        self.qs = []
        for i in range(self.n_vocabs):
            if self.verbose is True:
                print(f"Training vocab #{i+1}")
            idx = sample(range(len(X_mat)), int(2e5))
            if self.verbose is True:
                print(f"Training KMeans...")
            self.vocabs.append(KMeans(n_clusters=self.k).fit(X_mat[idx]))
            self.centers.append(self.vocabs[i].cluster_centers_)
            if self.lcs is True and self.norming == "RN":
                if self.verbose is True:
                    print("Finding rotation-matrices...")
                predicted = self.vocabs[i].predict(X_mat)
                qsi = []
                for j in range(self.k):
                    q = PCA(n_components=X_mat.shape[1]).fit(X_mat[predicted == j]).components_
                    # Below does not work everytime!
                    # print(f"shape predicted for center {j}: {X_mat[predicted == j].shape}; q.shape: {q.shape}")
                    qsi.append(q)
                self.qs.append(qsi)
        self.database = self._extract_vlads(X)
        return self

    def transform(self, X):
        """Transform the input-tensor to a matrix of VLAD-descriptors

        Parameters
        ----------
        X : list(array)
            List of image-descriptors

        Returns
        -------
        vlads : array, shape (n, d * self.k)
            The transformed VLAD-descriptors
        """
        vlads = self._extract_vlads(X)
        return vlads

    def fit_transform(self, X):
        """Fit the model and transform the input-data subsequently

        Parameters
        ----------
        X : list(array)
            List of image-descriptors

        Returns
        -------
        vlads : array, shape (n, d * self.k)
            The transformed VLAD-descriptors
        """
        _ = self.fit(X)
        vlads = self.transform(X)
        return vlads

    def refit(self, X):
        """Refit the Visual Vocabulary

        Uses the already learned cluster-centers as in initial values for
        the KMeans-models

        Parameters
        ----------
        X : array
            The database used to refit the visual vocabulary

        Returns
        -------
        self : VLAD
            Refitted object
        """
        self.vocabs = []
        self.centers = []

        for i in range(self.n_vocabs):
            self.vocabs.append(KMeans(n_clusters=self.k, init=self.centers).fit(X.transpose((2, 0, 1))
                                                                                .reshape(-1, X.shape[1])))
            self.centers.append(self.vocabs[i].cluster_centers_)

        self.database = self._extract_vlads(X)
        return self

    def predict(self, desc):
        """Predict class of given descriptor-matrix

        Parameters
        ----------
        desc : array
            A descriptor-matrix (m x d)

        Returns
        -------
        ``argmax(self.predict_proba(desc))`` : array
        """
        return np.argmax(self.predict_proba(desc))

    def predict_proba(self, desc):
        """Predict class of given descriptor-matrix, return probability

        Parameters
        ----------
        desc : array
            A descriptor-matrix (m x d)

        Returns
        -------
        ``self.database @ vlad``
            The similarity for all database-classes
        """
        vlad = self._vlad(desc)  # Convert to VLAD-descriptor
        if self.aggregate == 'concat':
            probas = self.database @ vlad  # Similarity between L2-normed vectors is defined as dot-product
        else:
            probas = np.einsum('ij,jik->ki', vlad, np.transpose(self.database, axes=(1, 0, 2))).mean(axis=1)
        return probas

    def _vlad(self, X):
        """Construct the actual VLAD-descriptor from a matrix of local descriptors

        Parameters
        ----------
        X : array
            Descriptor-matrix for a given image

        Returns
        -------
        ``V.flatten()`` : array
            The VLAD-descriptor
        """
        np.seterr(invalid='ignore', divide='ignore')  # Division with 0 encountered below
        vlads = []

        for j in range(self.n_vocabs):  # Compute for multiple vocabs
            predicted = self.vocabs[j].predict(X)
            m, d = X.shape
            V = np.zeros((self.k, d))  # Initialize VLAD-Matrix

            if self.norming == "RN":
                for i in range(self.k):
                    curr = X[predicted == i] - self.centers[j][i]
                    V[i] = np.sum(curr / norm(curr, axis=1)[:, None], axis=0)
                    if self.lcs is True:
                        V[i] = self.qs[j][i] @ V[i]  # Equivalent to multiplication in  summation above
            else:
                for i in range(self.k):
                    V[i] = np.sum(X[predicted == i] - self.centers[j][i], axis=0)

            if self.norming in ("intra", "RN"):
                V /= norm(V, axis=1)[:, None]  # L2-normalize every sum of residuals
                np.nan_to_num(V, copy=False)  # Some of the rows contain 0s. np.nan will be inserted when dividing by 0!

            if self.norming in ("original", "RN"):
                V = self._power_law_norm(V)

            V /= norm(V)  # Last L2-norming
            V = V.flatten()
            vlads.append(V)
        if self.aggregate == 'concat':
            vlads = np.concatenate(vlads)
            vlads /= norm(vlads)  # Not on axis, because already flat
        else:
            vlads = np.vstack(vlads)  # Return shape is (n_vocabs, len(vlad))
        return vlads

    def _extract_vlads(self, X):
        """Extract VLAD-descriptors for a number of images

        Parameters
        ----------
        X : list(array)
            List of image-descriptors

        Returns
        -------
        database : array
            Database of all VLAD-descriptors for the given Tensor
        """
        vlads = []
        for x in pb.progressbar(X):
            vlads.append(self._vlad(x))
        if self.aggregate == 'concat':
            database = np.vstack(vlads)
        else:
            database = np.dstack(vlads)
        return database

    def _add_to_database(self, vlad):
        """Add a given VLAD-descriptor to the database

        Parameters
        ----------
        vlad : array
            The VLAD-descriptor that should be added to the database

        Returns
        -------
        ``None``
        """
        self.database = np.vstack((self.database, vlad))

    def _power_law_norm(self, X):
        """Perform power-Normalization on a given array

        Parameters
        ----------
        X : array
            Array that should be normalized

        Returns
        -------
        normed : array
            Power-normalized array
        """
        normed = np.sign(X) * np.abs(X)**self.alpha
        return normed

    def __repr__(self):
        return f"VLAD(k={self.k}, norming=\"{self.norming}\")"

    def __str__(self):
        return f"VLAD(k={self.k}, norming=\"{self.norming}\")"

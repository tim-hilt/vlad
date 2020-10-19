import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans
from joblib import dump, load
import progressbar as pb


class VLAD:
    """VLAD - Vector of Locally Aggregated Descriptors

    This class provides an implementation of the original proposal of "Vector of Locally Aggregated Descriptors" (VLAD)
    originally proposed in [1]_.

    Parameters
    ----------
    k : int, default=256
        Number of clusters to obtain for the visual vocabulary.
    norming : {"original", "intra", "RN"}, default="original"
        How the norming of the VLAD-descriptors should be performed.
        For more info see below.

    Attributes
    ----------
    dictionary : sklearn.cluster.KMeans(k)
        The visual vocabulary of the object
    centers : array
        The centroids for the visual vocabulary
    database : array
        All known VLAD-vectors

    Notes
    -----
    ``norming="original"`` uses the original formulation of [1]_. An updated formulation based on [2]_
    is provided by ``norming="intra"``. Finally the best norming based on [3]_ is provided by ``norming="RN"``.

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
    """
    def __init__(self, k=256, norming="original"):
        self.k = k
        self.norming = norming
        self.dictionary = None
        self.centers = None
        self.database = None

    def fit(self, X, save=True):
        """Fit Visual Vocabulary

        Parameters
        ----------
        X : array
            Tensor of image descriptors (m x d x n)
        save : bool
            If ``True`` save fitted vocabulary to disk

        Returns
        -------
        self : VLAD
            Fitted object
        """
        # TODO: Maybe generalize better by passing a list of descriptors instead of a tensor!
        if self.dictionary is None:
            self.dictionary = KMeans(n_clusters=self.k).fit(X.transpose((2, 0, 1))
                                                            .reshape(-1, X.shape[1]))  # 3D to 2D
            self.centers = self.dictionary.cluster_centers_
            if save is True:
                _ = dump(self.dictionary, "dictionary.joblib")
        else:
            print("Dictionary already fitted. Use refit() to force retraining.")
        self.database = self._extract_vlads(X)
        return self

    def transform(self, X):
        """Transform the input-tensor to a matrix of VLAD-descriptors

        Parameters
        ----------
        X : array, shape (m, d, n)
            Tensor of image-descriptors

        Returns
        -------
        vlads : array, shape (n, d x self.k)
            The transformed VLAD-descriptors
        """
        vlads = self._extract_vlads(X)
        return vlads

    def fit_transform(self, X):
        """Fit the model and transform the input-data subsequently

        Parameters
        ----------
        X : array, shape (m, d, n)
            Tensor of image-descriptors

        Returns
        -------
        vlads : array, shape (n, d x self.k)
            The transformed VLAD-descriptors
        """
        _ = self.fit(X)
        vlads = self.transform(X)
        return vlads

    def refit(self, X, save=True):
        """Refit the Visual Vocabulary

        Parameters
        ----------
        X : array
            The database used to refit the visual vocabulary
        save : bool, default=True
            If ``True`` save refitted vocabulary to disk, possibly
            overwriting existing ``dictionary.joblib``

        Returns
        -------
        self : VLAD
            Refitted object
        """
        # TODO: Maybe generalize better by passing a list of descriptors instead of a tensor!
        self.dictionary = KMeans(n_clusters=self.k, init=self.centers).fit(X.transpose((2, 0, 1))
                                                                           .reshape(-1, X.shape[1]))
        self.centers = self.dictionary.cluster_centers_
        if save is True:
            _ = dump(self.dictionary, "dictionary.joblib")
        self.database = self._extract_vlads(X)
        return self

    def load_vocab(self, filename):
        """Manually load vocabulary from filename"""
        try:
            self.dictionary = load(str(filename))
            self.centers = self.dictionary.cluster_centers_
        except FileNotFoundError:
            print(f"The file {filename} was not found!")

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
        return self.database @ vlad  # Similarity between L2-normed vectors is defined as dot-product

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
        predicted = self.dictionary.predict(X)
        m, d = X.shape
        V = np.zeros((self.k, d))  # Initialize VLAD-Matrix

        if self.norming == "RN":
            for i in range(self.k):
                curr = X[predicted == i] - self.centers[i]
                V[i] = np.sum(curr / norm(curr, axis=1)[:, None], axis=0)
        else:
            for i in range(self.k):
                V[i] = np.sum(X[predicted == i] - self.centers[i], axis=0)  # TODO: Could this part be vectorized

        if self.norming in ("intra", "RN"):
            V /= norm(V, axis=1)[:, None]  # L2-normalize every sum of residuals
            np.nan_to_num(V, copy=False)  # Some of the rows contain 0s. np.nan will be inserted when dividing by 0!

        if self.norming in ("original", "RN"):
            V = self._power_law_norm(V)

        V /= norm(V)  # Last L2-norming

        return V.flatten()

    def _extract_vlads(self, X):
        """Extract VLAD-descriptors for a number of images

        Parameters
        ----------
        X : array, shape (m, d, n)
            Tensor of image-descriptors

        Returns
        -------
        database : array
            Database of all VLAD-descriptors for the given Tensor
        """
        vlads = []
        for i in pb.progressbar(range(X.shape[-1])):
            vlads.append(self._vlad(X[..., i]))
        database = np.vstack(vlads)
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

    def save_database(self):
        """Save the fitted database to disk

        Saving the database allows for future users to manually load it. This way some time can be
        saved and the object doesn't have to be fitted.

        Returns
        -------
        ``None``
        """
        if self.database is not None:
            np.save("database.npy", self.database)
        else:
            print("No database fitted yet. use fit() first.")

    def load_database(self, filename, force=False):
        """Manually load database

        Parameters
        ----------
        filename : str
            Filename of the database
        force : bool, default=True
            If `True` forces loading of the database, even if `self.database` is not None

        Returns
        -------
        ``None``
        """
        if self.database is None or force is True:
            self.database = np.load(filename)
        else:
            print("There's a database present already. Use force=True to overwrite it.")

    @staticmethod
    def _power_law_norm(X, alpha=.2):
        """Perform power-Normalization on a given array

        Parameters
        ----------
        X : array
            Array that should be normalized
        alpha : float, default=0.2
            The exponent for the root-part, default taken from [1]_

        Returns
        -------
        normed : array
            Power-normalized array

        References
        ----------
        .. [1] Delhumeau, J., Gosselin, P. H., Jégou, H., & Pérez, P. (2013, October).
               Revisiting the VLAD image representation. In Proceedings of the 21st ACM
               international conference on Multimedia (pp. 653-656).
        """
        normed = np.sign(X) * np.abs(X)**alpha
        return normed

    def __repr__(self):
        return f"VLAD(k={self.k}, norming=\"{self.norming}\")"

    def __str__(self):
        return f"VLAD(k={self.k}, norming=\"{self.norming}\")"

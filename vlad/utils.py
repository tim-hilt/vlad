import numpy as np
from numpy.linalg import norm


def RootSIFT(descs):
    """Convert usual SIFT-descriptors to RootSIFT as proposed in [1]

    Parameters
    ----------
    descs : array_like
        The SIFT-descriptors to adapt

    Returns
    -------
    descs : array_like
        Same type as input, output are the RootSIFT-converted descriptors

    Notes
    -----
    RootSIFT in relation to regular SIFT-descriptors consists in taking the square-root of
    the L1-normalized SIFT-descriptors [1]. It's very easy to implement and therefore
    demonstrated as a function in this repository.

    References
    ----------
    .. [1] Arandjelović, R., & Zisserman, A. (2012, June). Three things everyone
           should know to improve object retrieval. In 2012 IEEE Conference on Computer
           Vision and Pattern Recognition (pp. 2911-2918). IEEE.
    """
    if isinstance(descs, list):
        for i in range(len(descs)):
            descs[i] = np.sqrt(descs[i] / norm(descs, ord=1, axis=1))
    elif isinstance(descs, np.ndarray):
        descs = np.sqrt(descs / norm(descs, ord=1, axis=1)[:, np.newaxis])  # New axis in order to broadcast correctly
    else:
        print("{} not supported! Choose one of [list, numpy.ndarray].".format(type(descs)))
        return
    return descs

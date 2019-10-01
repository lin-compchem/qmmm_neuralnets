"""
This file contains methods for selecting feature from input vectors
"""
import numpy as np



def do_pca(bfs, n_components):
    """
    This is the code from my PCA experiment. Should do again and report
    results in the manual

    Params
    ------
    bfs: list of ndarrays
        The basis function arrays
    ncomponents:
        The number of principal components to keep
    
    Returns
    -------
    pca_models: list
        List of scikit-learn PCA models, required for back transformation

    Notes
    -----
    TODO: check this
    """
    from sklearn.decomposition import PCA
   
    pca_models = []
    for bf in bfs:
        pca = PCA(n_components=n_components, svd_solver='full')
        pca.fit(bf)
        pca_models.append(pca)

    return pca_models


def arbitrary_from_full_bpsf(h_rad, o_rad, h_ang, o_ang,
                             h_radg=None, o_radg=None, h_angg=None,
                             o_angg=None):
    """
    This method arbitrarily selects features from input vectors. The method
    is described in the example notebook: TODO
    This is kept here because it works well enough. The basis vectors must
    be the size of the tests in the FortBPSF package.

    Parameters
    ----------
    h_rad: ndarray of Nx48
    h_ang: ndarray of Nx36
    o_rad: ndarray of Nx48
    o_ang: ndarray of Nx54
    
    Returns
    -------
    hbf: ndarray
    obf: ndarray
    """
    grads = False
    assert h_rad.shape[1] == 48
    assert o_rad.shape[1] == 48
    assert h_ang.shape[1] == 36
    assert o_ang.shape[1] == 54

    if h_radg or o_radg or h_angg or o_angg:
        assert h_radg != 0
        assert o_radg != 0
        assert h_angg != 0
        assert o_angg != 0
        grads = True

    h_rx = [0, 24, 2, 25]
    h_ax = [1, 3, 7, 9, 19, 21, 23, 25, 27, 29, 31, 33]
    o_rx = [24, 25, 26, 0, 1, 2]
    o_ax = [23, 27, 36, 37, 38, 39, 42, 43, 44, 45, 48, 49, 50]
    hbf = np.float32(np.concatenate((np.delete(h_rad, h_rx, axis=1), np.delete(h_ang, h_ax, axis=1)), axis=1))
    obf = np.float32(np.concatenate((np.delete(o_rad, o_rx, axis=1), np.delete(o_ang, o_ax, axis=1)), axis=1))
    if grads:
        hg = np.float32(np.concatenate(
            (np.delete(h_radg, h_rx, axis=1), np.delete(h_radg, h_ax, axis=1)),
            axis=1))
        og = np.float32(np.concatenate(
            (np.delete(o_radg, o_rx, axis=1), np.delete(o_radg, o_ax, axis=1)),
            axis=1))
    h_rx = np.concatenate((np.arange(11, 22, 1), np.arange(31, 44, 1), (45, 46, 49, 50, 59, 60, 61, 62, 63, 64,65, 66, 67)))
    hbf = np.delete(hbf, h_rx, axis=1)
    if grads:
        hg = np.delete(hg, h_rx, axis=1)
    h_rx = (22, 23, 26, 28, 27)
    hbf = np.delete(hbf, h_rx, axis=1)
    if grads:
        hg = np.delete(hg, h_rx, axis=1)
    o_rx = np.concatenate((np.arange(9,21), np.arange(28,42), np.arange(43, 60, 2),
                              np.asarray((62, 63, 67, 71, 73, 76, 77, 78, 79, 80, 81, 82))))
    obf = np.delete(obf, o_rx, axis=1)
    if grads:
        og = np.delete(og, o_rx, axis=1)
    o_rx = [17, 19, 20, 21, 22, 23, 24, 25,29,31]
    obf = np.delete(obf, o_rx, axis=1)
    if grads:
        og = np.delete(og, o_rx, axis=1)
    if grads:
        return hbf, obf, hg, og
    return hbf, obf

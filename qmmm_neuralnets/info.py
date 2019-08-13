"""
This file contains subroutines which get information about the correction
or basis set files
"""
import h5py as h5
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from qmmm_neuralnets.files.bpsf_keys import *
from qmmm_neuralnets.files.h5_file_ops import *
from qmmm_neuralnets.units import AU2KCALPANG

def gradient_rms(grad_fname, natom_fname, natmk=NATMK, gradk=GRADK, au=False,
                 return_norms=False):
    """
    Returns the average RMS grad (rms grad/atom) for the h5file in filename

    Parameters
:   ----------
    grad_fname: str
        file name with gradients
    natom_fname: str
        file name with number of atoms
    natmk: str
        h5key for number of atoms
    gradk:str
        h5 key for gradient
    au: bool
        is the gradient in a.u.? if so convert to kcal/mol
    return_norms: bool
        return the gradient norms too.

    Returns
    -------
    info: dict
        Statistics about the gradient. Has the following keys:
            max_grad: float
                Note: this is the maximum absolute value of a cartesian grad
            mean_norm: float
            max_norm: float
            rms_grads: ndarray
    """
    check_for_keys(grad_fname, gradk)
    check_for_keys(natom_fname, natmk)
    info = {}

    # Load the data
    with h5.File(grad_fname, 'r') as ifi, h5.File(natom_fname, 'r') as afi:
        # The files must have the same first dimension
        try:
            assert(afi[natmk].shape[0] == ifi[gradk].shape[0])
        except AssertionError:
            sys.stderr.write("Error, files {} and {} must have same number of"
                             " geometries \n".format(natom_fname, grad_fname))
            raise
        # THe total number of atoms for the average
        natoms = afi[natmk][:]
        num_geoms = ifi[gradk].shape[0]
        grad = ifi[gradk][:]
    #
    # Convert au to kcal/mol
    if au:
        grad *= AU2KCALPANG

    # Get the RMS gradient of each atom in each molecule. We are calculating
    # for geoms which may not have atoms there
    grad = np.ma.masked_invalid(grad)
    info["max_grad"] = np.abs(grad).max()
    grad = np.linalg.norm(grad, axis=2)
    info["mean_norm"] = grad.mean()
    info["max_norm"] = grad.max()
    if return_norms:
        info["grad_norms"] = grad

    return info

def get_gradient_info(grad_fname, natom_fname):
    """
    Call multiple gradient information subroutines and print out their data

    Parameters
    ----------
    grad_fname: str
        h5 file with gradient info
    natom_fname: str
        h5 file with info for number of atoms

    Returns
    -------

    """
    def print_grad_info(grad_rms, max_rms, max_grad):
        print("The gradient RMS is (kcal/mol):                       {:10.6f}".format(grad_rms))
        print("The maximum atomic gradient norm is (kcal/mol):       {:10.6f}".format(max_rms))
        print("The maximum gradient in one dimension is (kcal/mol):  {:10.6f}".format(max_grad))

    check_for_keys(grad_fname, GRADK)
    cor_info = gradient_rms(grad_fname, natom_fname)
    se_info = gradient_rms(natom_fname, natom_fname, NATMK, SEGK)
    ref_info = gradient_rms(natom_fname, natom_fname, NATMK, REFGK, au=True)

    print("For correction gradient file:        {}".format(grad_fname))
    print_grad_info(cor_info["mean_norm"], cor_info["max_norm"],
            cor_info["max_grad"])
    print()
    print("For PM3 gradient file:               {}".format(natom_fname))
    print(print_grad_info(se_info["mean_norm"], se_info["max_norm"],
        se_info["max_grad"]))
    print()
    print("For REF gradient file:               {}".format(natom_fname))
    print(print_grad_info(ref_info["mean_norm"], ref_info["max_norm"],
        ref_info["max_grad"]))


def gradient_distplot():
    pass

def get_ener_info(ener_file, ener_key, return_energies=False):
    """
    Get energy info

    Parameters
    ---------
    eners: ndarray

    Returns
    -------
    info: dict
        Has keys:
            max_ener
    """
    info = {}
    check_for_keys(ener_file, ener_key)
    with h5.File(ener_file, 'r') as ifi:
        print('hi')
        ener = ifi[ener_key][:]
        info["max_ener"] = ener.max()
        info["min_ener"] = ener.min()
        info["max_absolute"] = np.abs(ener).max()
        info["mean_ener"] = ener.mean()
        info["ener_file"] = ener_file
        info["num_ener"] = ener.size
        if(return_energies):
            info["eners"] = ener
    return info

def print_ener_info(info):
    """
    Print the energy info
    """
    print("For file: {}".format(info["ener_file"]))
    print("    The number of energies is:       {}".format(info["num_ener"]))
    print("    The maximum energy is:           {}".format(info["max_ener"]))
    print("    The minimum energy is:           {}".format(info["min_ener"]))
    print("    The maximum absolute energy is:  {}".format(info["max_absolute"]))
    print("    The average energy is:           {}".format(info["mean_ener"]))
    return

def descriptive_ener_statistics(ener_file, ener_key, graph=False):
    info = get_ener_info(ener_file, ener_key, graph)
    print_ener_info(info)
    if graph:
        energy_dist_graph(info["eners"])

def gradient_dist_graph(grads, out_path):
    """
    This subroutine makes a distribution graph of the RMS gradients.
    It is currently not linked up into a package.abs

    Parameters
    ----------
    grads: ndarray
        The rms gradients to graph
    show: bool-like
        Call plt.show()?
    out_path: str
        Save the graph to this file

    """
    g = grads.flatten()
    g = g[~np.isnan(g)]
    figsize=(6,8.5)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    c = sns.color_palette()
    sns.set_context("paper")

    # First distribution plot with KDE
    img = sns.distplot(g, ax=axes[0])
    img.set_xlabel("Gradient RMS[kcal/mol]")
    img.set_ylabel("Density")

    # Second graph, but with counts instead of density estimate
    img = sns.distplot(g, ax=axes[1], hist=True, norm_hist=False, kde=False,
                       color=c[1])
    img.set_xlabel("Gradient RMS[kcal/mol]")
    img.set_ylabel("Count")

    if out_path:
        plt.savefig(out_path, figsize=figsize, dpi=300)
    return


def energy_dist_graph(eners, out_path):
    """
    This subroutine makes a distribution graph of the energies
    It is currently not linked up into a package.abs

    Parameters
    ----------
    grads: ndarray
        The rms gradients to graph
    show: bool-like
        Call plt.show()?
    out_path: str
        Save the graph to this file

    """
    figsize=(6,8.5)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    c = sns.color_palette()
    sns.set_context("paper")

    plt.suptitle("Energy Distributions\n"
                 "Mean of distribution {} subtracted from each energy".format(
        eners.mean()
    ))
    # First distribution plot with KDE
    img = sns.distplot(eners - eners.mean(), ax=axes[0])
    img.set_xlabel("Energy[kcal/mol]")
    img.set_ylabel("Density")

    # Second graph, but with counts instead of density estimate
    img = sns.distplot(eners - eners.mean(), ax=axes[1], hist=True,
            norm_hist=False,  kde=False, color=c[1])
    img.set_xlabel("Energy [kcal/mol]")
    img.set_ylabel("Count")

    if out_path:
        plt.savefig(out_path, figsize=figsize, dpi=300)
    return



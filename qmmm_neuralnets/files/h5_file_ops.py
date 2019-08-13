#!/usr/bin/env python
"""
Operations over generic quantities of H5 files
"""

import h5py as h5
import sys

def mean_over_files(fnames, key):
    """
    Calculate the mean of of a value over a list of h5 files

    Parameters
    ----------
    fnames: list of str
        list of hdf5 files with identical keys
    key: str
        Name of key to calculate mean over

    Returns
    -------
    mean: float or int
        The mean of the quantity
    """
    num_files = check_file_list(fnames)
    mean = 0
    for fname in fnames:
        with h5.File(fname, 'r') as ifi:
            try:
                mean += ifi[key][:].mean()
            except ValueError:
                raise
    return mean / num_files

def check_file_list(fnames):
    """
    Check to see if we have a list of files to do error-checking for various
    subroutines

    Parameters
    ----------
    fnames: list of str
        file names to check if list

    Returns
    -------
    num_files: int
        number of files in list

    """
    try:
        num_files = len(fnames)
        assert (num_files > 0)
    except ValueError:
        sys.stderr.write("Please pass a list of filenames as an argument")
        raise
    return num_files

def check_for_keys(fname, *keys):
    """
    Check if the key(s) exists in the h5 file

    Parameters
    ----------
    fname: str
        The name of the h5 file
    *keys:
        keys to check

    Returns
    -------
    """
    with h5.File(fname, 'r') as ifi:
        all_keys = list(ifi.keys())
        for key in keys:
            if key not in all_keys:
                sys.stderr.write("Error, key {} not in hdf5 file {}\n".format(
                    key, fname))
                raise KeyError

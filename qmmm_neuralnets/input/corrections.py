"""
File for input file preprocessing. This calculates the correction energy and
gradient for the terms
"""
import sys
import h5py as h5
from qmmm_neuralnets.files.bpsf_keys import *
from qmmm_neuralnets.files.h5_file_ops import *
from qmmm_neuralnets.units import *
import numpy as np

def water_correction_energies(fname, se_h2o_hof, se_h_hof, ref_h2o_ener,
                              se_au=False, ref_au=True):
    """
    Calculate the correction energies for water data.
    This function is used in the XSEDE 2019 proposal and related materials.

    Parameters
    ----------
    fname: str
        A list of filenames to read in the energy data from
    se_h2o_hof: float
        The heat of formation for water at the semiempirical QM method in
        kcal/mol
    se_h_hof: float
        The heat of formation for a proton at the semiempirical QM level in
        kcal/mol
    ref_h2o_ener: float
        The electronic energy for water at the reference QM level in eH
    se2au: bool
        Convert the semiempirical method energy from au to kcal?

    Returns
    -------

    Notes
    -----
    
    The correction is calculated via

    ..math::
    $E_{Corr} = E_{MP2} - \mu_{E_MP2} - (E_{SE} - \mu_{E_SE})$
    
    In order to predict future corrections in data sets, the constant
    
    $-\mu_{E_MP2} + \mu_{E_{SE}}$
    
    must be saved to reconstruct the same correction energies that the original
    network saw.
    
    This is stored in the module level variable "CORRECTION_OFFSET"
    
    
    The heat of formation for each correction energy is also used. This means
    that for a system of:
    ${H_7 O_3}_{HOF} = E_{H_7 O_3} - 3*E_{H_2 O} - E_{H^+}$
    Note that for Reference QM method, the electronic energy is used as the 
    HOF constant which makes E_{H^+} = 0.
    However, this is not the case for SE method, where the HOF is needed.
    """
    check_for_keys(fname, REFEK, NATMK, SEEK)
    with h5.File(fname, 'r') as ifi:
        # This calculates the reference heat of formation
        # Note the reference is assumed to be in eH
        correction = ifi[REFEK][:] - ((ifi[NATMK][:]//3) * ref_h2o_ener)
        if ref_au:
            correction *= 627.509
        if se_au:
            correction -= (ifi[SEEK][:] - se_h_hof - (ifi[NATMK][:]//3) * se_h2o_hof) * 627.509
        else:
            correction -= (ifi[SEEK][:] - se_h_hof - (ifi[NATMK][:]//3) * se_h2o_hof)
    return correction

def correction_gradients(fname, ref_au=True, nan_to_num=True, se_au=False):
    """
    Calculate the correction gradients for the file.

    The reference QM gradients are assumed to be in Eh, the SE in kcal

    Parameters
    ----------
    fname: str
        The name of the file to get the gradients from

    Returns
    -------

    """
    check_for_keys(fname, REFGK, SEGK)
    with h5.File(fname, 'r') as ifi:
        if ref_au:
            grad = ifi[REFGK][:] * AU2KCALPANG
        else:
            grad = ifi[REFGK][:]
        if se_au:
            grad -= ifi[SEGK][:] * AU2KCALPANG
        else:
            grad -= ifi[SEGK][:]
    if nan_to_num:
        np.nan_to_num(grad, copy=False)
    return grad


def write_water_correction_file(in_path, out_path, ener=True, grad=True,
                                se_h2o_hof=-53.10182,
                                se_h_hof=353.58586,
                                ref_h2o_ener=-76.3186321529223,
                                nan_to_num=True,
                                se_au=False):

    with h5.File(out_path, 'w') as ofi:
        ofi.create_dataset(ENERK, data=water_correction_energies(in_path,
                                se_h2o_hof, se_h_hof, ref_h2o_ener,
                                se_au=se_au))
        ofi.create_dataset(GRADK, data=correction_gradients(in_path,
                           nan_to_num=nan_to_num, se_au=se_au))


def calculate_basis_gather(fname, timing=10):

    check_for_keys(fname, NATMK, ANUMK)

    hi = 0
    oi = 0

    with h5.File(fname, 'r') as ifi:
        num_geoms = ifi[NATMK].size
        max_atoms = ifi[COORK].shape[-2]
        atm_ind = np.zeros(max_atoms)
        num_h = np.sum(ifi['atomic_number'][:] == 1)
        num_o = np.sum(ifi['atomic_number'][:] == 8)
        h_gather = np.zeros((num_h, max_atoms), dtype=np.int32)
        o_gather = np.zeros((num_o, max_atoms), dtype=np.int32)

        for g in range(num_geoms):
            if (g+1) % timing == 0:
                print("On geom: ", g + 1)
                break
            atm_ind[:] = 0
            start_h = hi
            start_o = oi
            for i in range(ifi['num_atoms'][g]):
                if ifi['atomic_number'][g, i] == 1:
                    atm_ind[i] = hi
                    hi += 1
                elif ifi['atomic_number'][g, i] == 8:
                    atm_ind[i] = oi
                    oi += 1
                else:
                    raise
            h_gather[start_h:hi] = atm_ind[:]
            o_gather[start_o:oi] = atm_ind[:]
    return h_gather, o_gather






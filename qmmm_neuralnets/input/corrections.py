"""
File for input file preprocessing. This calculates the correction energy and
gradient for the terms
"""
import sys
import h5py as h5
from qmmm_neuralnets.files.bpsf_keys import *
from qmmm_neuralnets.files.h5_file_ops import *
from qmmm_neuralnets.units import *


def water_correction_energies(fname, se_h2o_hof, se_h_hof, ref_h2o_ener):
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
        correction *= 627.509
        correction -= (ifi[SEEK][:] - se_h_hof - (ifi[NATMK][:]//3) * se_h2o_hof)
    return correction

def correction_gradients(fname, ref_au=True):
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
            return ifi[REFGK][:] * AU2KCALPANG - ifi[SEGK][:]
        else:
            return ifi[REFGK][:] - ifi[SEGK][:]


def write_water_correction_file(in_path, out_path, ener=True, grad=True,
                                se_h2o_hof=-53.10182,
                                se_h_hof=353.58586,
                                ref_h2o_ener=-76.3186321529223):

    with h5.File(out_path, 'w') as ofi:
        ofi.create_dataset(ENERK,data=water_correction_energies(in_path,
                                se_h2o_hof, se_h_hof, ref_h2o_ener))
        ofi.create_dataset(GRADK, data=correction_gradients(in_path))

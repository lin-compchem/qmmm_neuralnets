"""
This holds classes and methods for dealing with a BPSF dataset
"""
import h5py as h5
import numpy as np

from qmmm_neuralnets.files.bpsf_keys import *
from qmmm_neuralnets.files.h5_file_ops import check_for_keys
from qmmm_neuralnets.input.filter import arbitrary_from_full_bpsf

class HO_BPSF():
    """
    This class contains methods for working with BPSF inputs that only have O and H radial distribution functions

    These are used in the XSEDE 2019 proposal.
    """
    def __init__(self, correction_path, symfunc_path, bas='full'):
        check_for_keys(correction_path, ENERK)
        check_for_keys(symfunc_path, ORADK, HRADK, OANGK, HANGK, OB2MK, HB2MK)
        with h5.File(correction_path, 'r') as ifi:
            self.yy = ifi[ENERK][:]
        with h5.File(symfunc_path, 'r') as ifi:
            o_rad = ifi[ORADK][:]
            h_rad = ifi[HRADK][:]
            o_ang = ifi[OANGK][:]
            h_ang = ifi[HANGK][:]
            self.io = ifi[OB2MK][:]
            self.ih = ifi[HB2MK][:]
            if bas == 'full':
                self.hbf = (np.concatenate(h_rad, h_ang))
                self.obf = (np.concatenate(o_rad, o_ang))
            elif bas == 'arb': # This is the arbitrary basis function
                self.hbf, self.obf =\
                    arbitrary_from_full_bpsf(h_rad, o_rad, h_ang, o_ang) 
        return

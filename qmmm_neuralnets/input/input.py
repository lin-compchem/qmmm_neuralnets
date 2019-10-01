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
    def __init__(self, correction_path, symfunc_path, bas='full', grad=False,
                 nan=True):
        check_for_keys(correction_path, ENERK)
        check_for_keys(symfunc_path, ORADK, HRADK, OANGK, HANGK, OB2MK, HB2MK)
        with h5.File(correction_path, 'r') as ifi:
            self.yy = ifi[ENERK][:]
            if grad:
                self.ygrad = ifi[CGK][:]
        with h5.File(symfunc_path, 'r') as ifi:
            o_rad = ifi[ORADK][:]
            h_rad = ifi[HRADK][:]
            o_ang = ifi[OANGK][:]
            h_ang = ifi[HANGK][:]
            self.io = ifi[OB2MK][:]
            self.ih = ifi[HB2MK][:]
            self.m2b = ifi[M2BK][:]
            if grad:
                h_radg = ifi[HRGK]
                o_radg = ifi[ORGK]
                h_angg = ifi[HAGK]
                o_angg = ifi[OAGK]
            if bas == 'full':
                self.hbf = np.concatenate((h_rad, h_ang), axis=1)
                self.obf = np.concatenate((o_rad, o_ang), axis=1)
                if grad:
                    self.hbg = np.concatenate((h_radg, h_angg), axis=1)
                    self.obg = np.concatenate((o_radg, o_angg), axis=1)
            elif bas == 'arb': # This is the arbitrary basis function
                if grad:
                    self.hbf, self.obf, self.hbg, self.obg =\
                        arbitrary_from_full_bpsf(h_rad, o_rad, h_ang, o_ang,
                                                 h_radg, o_radg, h_angg, o_angg)
                else:
                    self.hbf, self.obf =\
                        arbitrary_from_full_bpsf(h_rad, o_rad, h_ang, o_ang)
        return

    def gen_data(self, batch_size=1):
        """
        This is a generator that allows one to get out some data from the
        basis sets in this input structure

        Parameters
        ----------
        hbf
        obf
        ihh
        ioo
        m2b
        en
        batch_size

        Returns
        -------

        """
        num_molecules = self.m2b.shape[0]
        i = 0
        j = i + batch_size
        assert j > 0
        while i < num_molecules:
            if j >= num_molecules:
                j = num_molecules - 1
            xh = self.hbf[self.m2b[i, 0, 0]:self.m2b[j, 0, 0]]
            xo = self.obf[self.m2b[i, 1, 0]:self.m2b[j, 1, 0]]
            ih = self.ihh[self.m2b[i, 0, 0]:self.m2b[j, 0, 0]] - i
            io = self.ioo[self.m2b[i, 1, 0]:self.m2b[j, 1, 0]] - i
            y = self.yy[i:j]
            yield xh, xo, ih, io, y
            i += batch_size
            if i > num_molecules:
                return

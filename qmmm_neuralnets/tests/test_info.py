#!/usr/bin/env python
import pytest
from numpy import isclose
import sys
import h5py as h5

import qmmm_neuralnets as qmnn
from qmmm_neuralnets.files.bpsf_keys import *

def test_qmmm_neuralnets_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qmmm_neuralnets" in sys.modules


def test_gradient_info():
    """ Test gradient info subroutine"""
    qmnn.info.get_gradient_info("../data/h9o4_corrections.h5",
           "../data/h9o4.h5")

def test_gradient_rms():
    """ Test gradient rms subroutine"""
    info = qmnn.info.gradient_rms("../data/h9o4_corrections.h5",
           "../data/h9o4.h5")
    assert isclose(info["mean_norm"], 66.63582583950173)
    assert isclose(info["max_norm"], 242.05346068025224)
    assert isclose(info["max_grad"], 231.38814784133947)

def test_energy_info():
    """ Test energy info subroutine"""
    qmnn.info.descriptive_ener_statistics("../data/h9o4_corrections.h5",ENERK)

def test_gradient_plot():
    data = qmnn.info.gradient_rms("../data/h9o4_corrections.h5",
            "../data/h9o4.h5", return_norms=True)
    qmnn.info.gradient_dist_graph(data['grad_norms'], "grads.svg")

def test_energy_plot():
    with h5.File("../data/h9o4_corrections.h5") as ifi:
        eners = ifi[ENERK][:]
    qmnn.info.energy_dist_graph(eners, "eners.svg")


test_energy_plot()

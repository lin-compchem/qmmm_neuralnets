#!/usr/bin/env python
import pytest
import qmmm_neuralnets as qmnn
import sys
from numpy import isclose

def test_qmmm_neuralnets_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qmmm_neuralnets" in sys.modules


def test_correction_energies():
    if_path = "../data/h9o4.h5"
    se_h2o_hof = -53.10182
    se_h_hof = 353.58586
    ref_h2o_ener = -76.3186321529223
    answer = 601.4361969282863
    energies = qmnn.input.water_correction_energies("../data/h9o4.h5",
                                                    se_h2o_hof,
                                                    se_h_hof,
                                                    ref_h2o_ener)
    assert isclose(answer, float(energies[0]))


def test_correction_gradients():
    grads = qmnn.input.correction_gradients("../data/h9o4.h5")
    print(grads[500])



"""
This file contains functions for generating test data from other test data
"""
def create_h9o4_corrections():
    """ This subroutine creates the water example test data for other tests"""
    qmnn.input.write_water_correction_file("../data/h9o4.h5",
            "../data/h9o4_corrections.h5")

create_h9o4_corrections()

"""
Unit and regression test for the qmmm_neuralnets package.
"""

# Import package, test suite, and other packages as needed
import qmmm_neuralnets
import pytest
import sys

def test_qmmm_neuralnets_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "qmmm_neuralnets" in sys.modules

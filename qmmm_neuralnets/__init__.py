"""
qmmm_neuralnets
This contains neural network creation and training programs for making networks compatible with the Fort-BPSF interface.
"""

# Add imports here
#from .qmmm_neuralnets import *
import qmmm_neuralnets.files
import qmmm_neuralnets.input
import qmmm_neuralnets.units
import qmmm_neuralnets.info
import qmmm_neuralnets.layers
#import .models

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

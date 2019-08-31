###############################################################################
"""Framework to build a smart, low-computational noise filter.
``noize`` offers low-computational algorithms for training
deep learning models as noise classifiers as well as for low-computational
noise filters.
"""
from .file_architecture import paths
from .file_architecture.paths import PathSetup
from .acousticfeats_ml.featorg import audio2datasets
from .acousticfeats_ml.modelfeats import PrepFeatures
from .acousticfeats_ml.modelfeats import prepfeatures as run_featprep 
from .acousticfeats_ml.modelfeats import loadfeature_settings as getfeatsettings
from .filterfun import filters
from .filterfun.applyfilter import filtersignal
from .mathfun import dsp, matrixfun


__all__=['paths', 'PathSetup', 'audio2datasets', 'PrepFeatures', 'run_featprep',
         'getfeatsettings','filters', 'dsp', 'matrixfun', 'filtersignal']

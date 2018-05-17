"""Wraps all Mantid algorithms so they use mslice's wrapped workspaces"""

from mantid.simpleapi import * # noqa: F401
from mslice.util.mantid.algorithm_wrapper import wrap_algorithm
from mantid.api import AlgorithmFactory

algorithms = AlgorithmFactory.getRegisteredAlgorithms(False)

for algorithm in algorithms.keys():
    globals()[algorithm] = wrap_algorithm(globals()[algorithm])

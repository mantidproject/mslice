"""Wraps all Mantid algorithms so they use mslice's wrapped workspaces"""

from mantid.simpleapi import * # noqa: F401
from mslice.util.mantid.algorithm_wrapper import run_algorithm_2
from mantid.api import AlgorithmFactory

algorithms = AlgorithmFactory.getRegisteredAlgorithms(False)

for algorithm in algorithms.keys():
    globals()[algorithm] = run_algorithm_2(globals()[algorithm])

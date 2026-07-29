"""Wraps all Mantid algorithms so they use mslice's wrapped workspaces"""

from mantid.api import AlgorithmFactory, AlgorithmManager
from mantid.simpleapi import *
from mantid.simpleapi import _create_algorithm_function

from mslice.util.mantid.algorithm_wrapper import wrap_algorithm

algorithms = AlgorithmFactory.getRegisteredAlgorithms(False)

for algorithm, versions in algorithms.items():
    try:
        globals()[algorithm] = wrap_algorithm(globals()[algorithm])
    except KeyError:  # Possibly a user defined algorithm
        try:
            alg_obj = AlgorithmManager.createUnmanaged(algorithm, max(versions))
            alg_obj.initialize()
        except Exception:  # noqa: S110
            pass
        else:
            globals()[algorithm] = wrap_algorithm(
                _create_algorithm_function(algorithm, max(versions), alg_obj)
            )

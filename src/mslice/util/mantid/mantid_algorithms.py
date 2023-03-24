"""Wraps all Mantid algorithms so they use mslice's wrapped workspaces"""

from mantid.simpleapi import *  # noqa: F401, F403
from mantid.simpleapi import _create_algorithm_function
from mslice.util.mantid.algorithm_wrapper import wrap_algorithm
from mantid.api import AlgorithmFactory, AlgorithmManager
from six import iteritems

algorithms = AlgorithmFactory.getRegisteredAlgorithms(False)

for algorithm, versions in iteritems(algorithms):
    try:
        globals()[algorithm] = wrap_algorithm(globals()[algorithm])
    except KeyError:   # Possibly a user defined algorithm
        try:
            alg_obj = AlgorithmManager.createUnmanaged(algorithm, max(versions))
            alg_obj.initialize()
        except Exception:
            pass
        else:
            globals()[algorithm] = wrap_algorithm(_create_algorithm_function(algorithm, max(versions), alg_obj))

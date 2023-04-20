import unittest

from mslice.models.alg_workspace_ops import get_number_of_steps
from mslice.models.axis import Axis


class AlgWorkspaceOpsTest(unittest.TestCase):

    def test_get_number_of_steps_returns_one_when_step_equals_zero(self):
        axis = Axis('DeltaE', 3, 33, 0)

        n_steps = get_number_of_steps(axis)
        self.assertEqual(1, n_steps)

    def test_get_number_of_steps_for_simple_division(self):
        axis = Axis('DeltaE', 3, 33, 2)

        n_steps = get_number_of_steps(axis)
        self.assertEqual(15, n_steps)

    def test_get_number_of_steps_for_remainder_division_to_floor(self):
        axis = Axis('DeltaE', 3, 33.03, 2)

        n_steps = get_number_of_steps(axis)
        self.assertEqual(15, n_steps)

    def test_get_number_of_steps_for_remainder_division_to_ceiling(self):
        axis = Axis('DeltaE', 3, 34, 2)

        n_steps = get_number_of_steps(axis)
        self.assertEqual(16, n_steps)

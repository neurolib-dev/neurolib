import unittest
import numpy as np
from neurolib.optimal_control.oc import solve_adjoint, update_control_with_limit


class TestOC(unittest.TestCase):
    """
    Test functions in neurolib/optimal_control/oc.py
    """

    @staticmethod
    def get_arbitrary_array_finite_values():
        """2x2x10 array filled with arbitrary positive and negative values in range [-5.0, 5.0]."""
        return np.array(
            [
                [
                    [1.0, 2.0, 0.0, -1.5123, 2.35, 1.0, -1.0, 5.0, 0.0, 0.0],
                    [-1.0, 3.0, 2.0, 0.5, -0.1, -0.2, 0.2, -0.2, 0.1, 0.5],
                ],
                [
                    [-0.5, 0.5, 0.5, -0.1, -1.0, 2.05, 3.1, -4.0, -5.0, -0.7],
                    [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 2.5],
                ],
            ]
        )

    @staticmethod
    def get_arbitrary_array():
        """2x2x10 array filled with arbitrary positive and negative values and +/-np.inf. Besides +/-np.inf all values
        fall in the range [-5., 5.]."""
        return np.array(
            [
                [
                    [1.0, 2.0, 0.0, -1.5123, 2.35, 1.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.0, 2.0, 0.5, -np.inf, np.inf, 0.2, -0.2, 0.1, 5.0],
                ],
                [
                    [-0.5, np.inf, 0.5, -0.1, -1.0, 2.5, 3.01, -4.0, -5.0, -0.7],
                    [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 2.5],
                ],
            ]
        )

    def test_solve_adjoint(self):
        # ToDo: Implement test for the solve_adjoint-function.
        pass

    def test_update_control_with_limit_no_limit(self):
        # Test for the control to be unchanged, if no limit is set.
        control = self.get_arbitrary_array_finite_values()
        step = 1.0
        cost_gradient = self.get_arbitrary_array()
        u_max = None

        control_limited = update_control_with_limit(control, step, cost_gradient, u_max)

        self.assertTrue(np.all(control_limited == control + step * cost_gradient))

    def test_update_control_with_limit_limited(self):
        # Test that absolute value of control signal is limited.
        control = self.get_arbitrary_array_finite_values()
        step = 1.0
        cost_gradient = self.get_arbitrary_array()
        u_max = 5.0

        control_limited = update_control_with_limit(control, step, cost_gradient, u_max)

        self.assertTrue(np.all(np.abs(control_limited) <= u_max))

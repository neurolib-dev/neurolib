import unittest
from neurolib.optimal_control import cost_functions
import numpy as np


class TestCostFunctions(unittest.TestCase):
    @staticmethod
    def get_arbitrary_array():
        return np.array([[1, -10, 5.555], [-1, 3.3333, 9]])  # an arbitrary vector with positive and negative entries

    def test_precision_cost_full_timeseries(self):
        w_p = 1
        x_target = self.get_arbitrary_array()

        self.assertEqual(cost_functions.precision_cost(x_target, x_target, w_p), 0)  # target and simulation coincide

        x_sim = np.copy(x_target)
        x_sim[:, 0] = -x_sim[:, 0]  # create setting where result depends only on this first entries
        self.assertEqual(cost_functions.precision_cost(x_target, x_target, w_p), 0)
        self.assertEqual(
            cost_functions.precision_cost(x_target, x_sim, w_p),
            w_p / 2 * np.sum((2 * x_target[:, 0]) ** 2),
        )

    def test_energy_cost(self):
        reference_result = 112.484456945
        w_2 = 1
        u = self.get_arbitrary_array()
        energy_cost = cost_functions.energy_cost(u, w_2)
        self.assertEqual(energy_cost, reference_result)

    def test_derivative_precision_cost_full_timeseries(self):
        w_p = 1
        x_target = self.get_arbitrary_array()
        x_sim = np.copy(x_target)
        x_sim[:, 0] = -x_sim[:, 0]  # create setting where result depends only on this first entries

        derivative_p_c = cost_functions.derivative_precision_cost(x_target, x_sim, w_p)

        self.assertTrue(np.all(derivative_p_c[:, 1::] == 0))
        self.assertTrue(np.all(derivative_p_c[:, 0] == 2 * (-w_p * x_target[:, 0])))

    def test_derivative_energy_cost(self):
        w_e = -0.9995
        u = self.get_arbitrary_array()
        desired_output = w_e * u
        self.assertTrue(np.all(cost_functions.derivative_energy_cost(u, w_e) == desired_output))

    def test_precision_cost_in_interval(self):
        """This test is analogous to the 'test_precision_cost'. However, the signal is repeated twice, but only
        the second interval is to be taken into account.
        """
        w_p = 1
        x_target = np.hstack((self.get_arbitrary_array(), self.get_arbitrary_array()))
        x_sim = np.copy(x_target)
        x_sim[:, 3] = -x_sim[:, 3]
        interval = (3, None)
        precision_cost = cost_functions.precision_cost(x_target, x_sim, w_p, interval)
        # Result should only depend on second half of the timeseries.
        self.assertEqual(precision_cost, w_p / 2 * np.sum((2 * x_target[:, 3]) ** 2))

    def test_derivative_precision_cost_in_interval(self):
        """This test is analogous to the 'test_derivative_precision_cost'. However, the signal is repeated twice, but
        only the second interval is to be taken into account.
        """
        w_p = 1
        x_target = np.hstack((self.get_arbitrary_array(), self.get_arbitrary_array()))
        x_sim = np.copy(x_target)
        x_sim[:, 3] = -x_sim[:, 3]  # create setting where result depends only on this first entries
        interval = (3, None)
        derivative_p_c = cost_functions.derivative_precision_cost(x_target, x_sim, w_p, interval)

        self.assertTrue(np.all(derivative_p_c[:, 0:3] == 0))
        self.assertTrue(np.all(derivative_p_c[:, 4::] == 0))
        self.assertTrue(np.all(derivative_p_c[:, 3] == 2 * (-w_p * x_target[:, 3])))


if __name__ == "__main__":
    unittest.main()

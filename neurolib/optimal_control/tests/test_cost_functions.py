import unittest
from neurolib.optimal_control import cost_functions
import numpy as np


class TestCostFunctions(unittest.TestCase):

    @staticmethod
    def get_arbitrary_array(self):
        return np.array([1, -10, 5.555])  # an arbitrary vector with positive and negative entries

    def test_precision_cost(self):
        w_p = 1
        x_target = self.get_arbitrary_array(self)

        self.assertEqual(cost_functions.precision_cost(x_target, x_target, w_p), 0)     # target and simulation coincide

        x_sim = np.copy(x_target)
        x_sim[0] = - x_sim[0]   # create setting where result depends only on this first entry
        self.assertEqual(cost_functions.precision_cost(x_target, x_target, w_p), 0)
        self.assertEqual(cost_functions.precision_cost(x_target, x_sim, w_p), w_p/2*(2*x_sim[0])**2)

    def test_energy_cost(self):
        w_2 = 1
        u = self.get_arbitrary_array(self)
        energy_cost = cost_functions.energy_cost(u, w_2)
        self.assertEqual(energy_cost, 65.9290125)

    def test_derivative_precision_cost(self):
        # ToDo
        pass

    def test_derivative_energy_cost(self):
        w_e = -0.9995
        u = self.get_arbitrary_array(self)
        desired_output = w_e*u
        self.assertTrue(np.all(cost_functions.derivative_energy_cost(u, w_e) == desired_output))


if __name__ == '__main__':
    unittest.main()

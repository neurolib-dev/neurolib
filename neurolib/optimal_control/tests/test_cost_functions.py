import unittest
from neurolib.optimal_control import cost_functions
import numpy as np


class TestCostFunctions(unittest.TestCase):
    @staticmethod
    def get_arbitrary_array():
        return np.array([[1, -10, 5.555], [-1, 3.3333, 9]])[
            np.newaxis, :, :
        ]  # an arbitrary vector with positive and negative entries

    def test_precision_cost_full_timeseries(self):
        print(" Test precision cost full timeseries")
        w_p = 1
        N = 1
        precision_cost_matrix = np.ones((N, 2))
        dt = 0.1
        x_target = self.get_arbitrary_array()
        interval = (0, x_target.shape[2])

        self.assertEqual(
            cost_functions.precision_cost(x_target, x_target, w_p, precision_cost_matrix, dt, interval),
            0,
        )  # target and simulation coincide

        x_sim = np.copy(x_target)
        x_sim[:, 0] = -x_sim[:, 0]  # create setting where result depends only on this first entries
        self.assertEqual(
            cost_functions.precision_cost(x_target, x_target, w_p, precision_cost_matrix, dt, interval),
            0,
        )
        self.assertEqual(
            cost_functions.precision_cost(x_target, x_sim, w_p, precision_cost_matrix, dt, interval),
            w_p / 2 * np.sum((2 * x_target[:, 0]) ** 2) * dt,
        )

    def test_precision_cost_nodes_channels(self):
        print(" Test precision cost full timeseries for node and channel selection.")
        w_p = 1
        N = 2
        x_target0 = self.get_arbitrary_array()
        x_target1 = 2.0 * self.get_arbitrary_array()
        target = np.concatenate((x_target0, x_target1), axis=0)
        precision_cost_matrix = np.zeros((N, 2))
        dt = 0.1
        interval = (0, target.shape[2])
        zerostate = np.zeros((target.shape))

        self.assertEqual(
            cost_functions.precision_cost(target, zerostate, w_p, precision_cost_matrix, dt, interval),
            0.0,
        )  # no cost if precision matrix is zero

        for i in range(N):
            for j in range(N):
                precision_cost_matrix[i, j] = 1
                result = w_p * 0.5 * sum((target[i, j, :] ** 2)) * dt
                self.assertEqual(
                    cost_functions.precision_cost(target, zerostate, w_p, precision_cost_matrix, dt, interval),
                    result,
                )
                precision_cost_matrix[i, j] = 0

    def test_derivative_precision_cost_full_timeseries(self):
        print(" Test precision cost derivative full timeseries")
        w_p = 1
        N = 1
        precision_cost_matrix = np.ones((N, 2))
        x_target = self.get_arbitrary_array()
        x_sim = np.copy(x_target)
        x_sim[0, :, 0] = -x_sim[0, :, 0]  # create setting where result depends only on this first entries
        interval = (0, x_target.shape[2])

        derivative_p_c = cost_functions.derivative_precision_cost(x_target, x_sim, w_p, precision_cost_matrix, interval)

        self.assertTrue(np.all(derivative_p_c[0, :, 1::] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 0] == 2 * (-w_p * x_target[0, :, 0])))

    def test_derivative_precision_cost_full_timeseries_nodes_channels(self):
        print(" Test precision cost derivative full timeseries for node and channel selection")
        w_p = 1
        N = 2
        precision_cost_matrix = np.ones((N, 2))
        x_target0 = self.get_arbitrary_array()
        x_target1 = 2.0 * self.get_arbitrary_array()
        target = np.concatenate((x_target0, x_target1), axis=0)
        precision_cost_matrix = np.zeros((N, 2))  # ToDo: overwrites previous definition, bug?
        zerostate = np.zeros((target.shape))
        interval = (0, target.shape[2])

        derivative_p_c = cost_functions.derivative_precision_cost(
            target, zerostate, w_p, precision_cost_matrix, interval
        )
        self.assertTrue(np.all(derivative_p_c == 0))

        for i in range(N):
            for j in range(N):
                precision_cost_matrix[i, j] = 1
                derivative_p_c = cost_functions.derivative_precision_cost(
                    target, zerostate, w_p, precision_cost_matrix, interval
                )
                result = -w_p * np.einsum("ijk,ij->ijk", target, precision_cost_matrix)
                self.assertTrue(np.all(derivative_p_c - result == 0))
                precision_cost_matrix[i, j] = 0

    def test_precision_cost_in_interval(self):
        """This test is analogous to the 'test_precision_cost'. However, the signal is repeated twice, but only
        the second interval is to be taken into account.
        """
        print(" Test precision cost in time interval")
        w_p = 1
        N = 1
        precision_cost_matrix = np.ones((N, 2))
        dt = 0.1
        x_target = np.concatenate((self.get_arbitrary_array(), self.get_arbitrary_array()), axis=2)
        x_sim = np.copy(x_target)
        x_sim[0, :, 3] = -x_sim[0, :, 3]
        interval = (3, x_target.shape[2])
        precision_cost = cost_functions.precision_cost(x_target, x_sim, w_p, precision_cost_matrix, dt, interval)
        # Result should only depend on second half of the timeseries.
        self.assertEqual(precision_cost, w_p / 2 * np.sum((2 * x_target[0, :, 3]) ** 2) * dt)

    def test_derivative_precision_cost_in_interval(self):
        """This test is analogous to the 'test_derivative_precision_cost'. However, the signal is repeated twice, but
        only the second interval is to be taken into account.
        """
        print(" Test precision cost derivative in time interval")
        w_p = 1
        N = 1
        precision_cost_matrix = np.ones((N, 2))
        x_target = np.concatenate((self.get_arbitrary_array(), self.get_arbitrary_array()), axis=2)
        x_sim = np.copy(x_target)
        x_sim[0, :, 3] = -x_sim[0, :, 3]  # create setting where result depends only on this first entries
        interval = (3, x_target.shape[2])
        derivative_p_c = cost_functions.derivative_precision_cost(x_target, x_sim, w_p, precision_cost_matrix, interval)

        self.assertTrue(np.all(derivative_p_c[0, :, 0:3] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 4::] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 3] == 2 * (-w_p * x_target[0, :, 3])))

    def test_energy_cost(self):
        print(" Test energy cost")
        dt = 0.1
        reference_result = 112.484456945 * dt
        w_2 = 1
        u = self.get_arbitrary_array()
        energy_cost = cost_functions.energy_cost(u, w_2, dt)
        self.assertEqual(energy_cost, reference_result)

    def test_derivative_energy_cost(self):
        print(" Test energy cost derivative")
        w_e = -0.9995
        u = self.get_arbitrary_array()
        desired_output = w_e * u
        self.assertTrue(np.all(cost_functions.derivative_energy_cost(u, w_e) == desired_output))


if __name__ == "__main__":
    unittest.main()

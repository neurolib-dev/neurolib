import unittest
import numpy as np
from neurolib.control.optimal_control import cost_functions
from neurolib.control.optimal_control.oc import getdefaultweights
from neurolib.utils.stimulus import ZeroInput
from neurolib.models.fhn import FHNModel
from neurolib.control.optimal_control import oc_fhn


class TestCostFunctions(unittest.TestCase):
    @staticmethod
    def get_arbitrary_array():
        return np.array([[1, -10, 5.555], [-1, 3.3333, 9]])[
            np.newaxis, :, :
        ]  # an arbitrary vector with positive and negative entries

    def test_precision_cost_full_timeseries(self):
        print(" Test precision cost full timeseries")
        N = 1
        precision_cost_matrix = np.ones((N, 2))
        dt = 0.1
        x_target = self.get_arbitrary_array()
        interval = (0, x_target.shape[2])
        weights = getdefaultweights()

        self.assertAlmostEqual(
            cost_functions.accuracy_cost(x_target, x_target, weights, precision_cost_matrix, dt, interval), 0, places=8
        )  # target and simulation coincide

        x_sim = np.copy(x_target)
        x_sim[:, 0] = -x_sim[:, 0]  # create setting where result depends only on this first entries
        self.assertAlmostEqual(
            cost_functions.accuracy_cost(x_target, x_target, weights, precision_cost_matrix, dt, interval), 0, places=8
        )
        self.assertAlmostEqual(
            cost_functions.accuracy_cost(x_target, x_sim, weights, precision_cost_matrix, dt, interval),
            weights["w_p"] / 2 * np.sum((2 * x_target[:, 0]) ** 2) * dt,
            places=8,
        )

    def test_precision_cost_nodes_channels(self):
        print(" Test precision cost full timeseries for node and channel selection.")
        N = 2
        x_target0 = self.get_arbitrary_array()
        x_target1 = 2.0 * self.get_arbitrary_array()
        target = np.concatenate((x_target0, x_target1), axis=0)
        precision_cost_matrix = np.zeros((N, 2))
        dt = 0.1
        interval = (0, target.shape[2])
        zerostate = np.zeros((target.shape))
        weights = getdefaultweights()

        self.assertAlmostEqual(
            cost_functions.accuracy_cost(target, zerostate, weights, precision_cost_matrix, dt, interval),
            0.0,
            places=8,
        )  # no cost if precision matrix is zero

        for i in range(N):
            for j in range(N):
                precision_cost_matrix[i, j] = 1
                result = weights["w_p"] * 0.5 * sum((target[i, j, :] ** 2)) * dt
                self.assertAlmostEqual(
                    cost_functions.accuracy_cost(target, zerostate, weights, precision_cost_matrix, dt, interval),
                    result,
                    places=8,
                )
                precision_cost_matrix[i, j] = 0

    def test_derivative_precision_cost_full_timeseries(self):
        print(" Test precision cost derivative full timeseries")
        weights = getdefaultweights()
        N = 1
        precision_cost_matrix = np.ones((N, 2))
        x_target = self.get_arbitrary_array()
        x_sim = np.copy(x_target)
        x_sim[0, :, 0] = -x_sim[0, :, 0]  # create setting where result depends only on this first entries
        interval = (0, x_target.shape[2])

        derivative_p_c = cost_functions.derivative_accuracy_cost(
            x_target, x_sim, weights, precision_cost_matrix, interval
        )

        self.assertTrue(np.all(derivative_p_c[0, :, 1::] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 0] == 2 * (weights["w_p"] * x_target[0, :, 0])))

    def test_derivative_precision_cost_full_timeseries_nodes_channels(self):
        print(" Test precision cost derivative full timeseries for node and channel selection")
        weights = getdefaultweights()
        N = 2
        x_target0 = self.get_arbitrary_array()
        x_target1 = 2.0 * self.get_arbitrary_array()
        target = np.concatenate((x_target0, x_target1), axis=0)
        precision_cost_matrix = np.zeros((N, 2))  # ToDo: overwrites previous definition, bug?
        zerostate = np.zeros((target.shape))
        interval = (0, target.shape[2])

        derivative_p_c = cost_functions.derivative_accuracy_cost(
            target, zerostate, weights, precision_cost_matrix, interval
        )
        self.assertTrue(np.all(derivative_p_c == 0))

        for i in range(N):
            for j in range(N):
                precision_cost_matrix[i, j] = 1
                derivative_p_c = cost_functions.derivative_accuracy_cost(
                    target, zerostate, weights, precision_cost_matrix, interval
                )
                result = weights["w_p"] * np.einsum("ijk,ij->ijk", target, precision_cost_matrix)
                self.assertTrue(np.all(derivative_p_c - result == 0))
                precision_cost_matrix[i, j] = 0

    def test_precision_cost_in_interval(self):
        """This test is analogous to the 'test_precision_cost'. However, the signal is repeated twice, but only
        the second interval is to be taken into account.
        """
        print(" Test precision cost in time interval")
        N = 1
        precision_cost_matrix = np.ones((N, 2))
        dt = 0.1
        x_target = np.concatenate((self.get_arbitrary_array(), self.get_arbitrary_array()), axis=2)
        x_sim = np.copy(x_target)
        x_sim[0, :, 3] = -x_sim[0, :, 3]
        interval = (3, x_target.shape[2])
        weights = getdefaultweights()
        precision_cost = cost_functions.accuracy_cost(x_target, x_sim, weights, precision_cost_matrix, dt, interval)
        # Result should only depend on second half of the timeseries.
        self.assertEqual(precision_cost, weights["w_p"] / 2 * np.sum((2 * x_target[0, :, 3]) ** 2) * dt)

    def test_derivative_precision_cost_in_interval(self):
        """This test is analogous to the 'test_derivative_precision_cost'. However, the signal is repeated twice, but
        only the second interval is to be taken into account.
        """
        print(" Test precision cost derivative in time interval")
        weights = getdefaultweights()
        N = 1
        precision_cost_matrix = np.ones((N, 2))
        x_target = np.concatenate((self.get_arbitrary_array(), self.get_arbitrary_array()), axis=2)
        x_sim = np.copy(x_target)
        x_sim[0, :, 3] = -x_sim[0, :, 3]  # create setting where result depends only on this first entries
        interval = (3, x_target.shape[2])
        derivative_p_c = cost_functions.derivative_accuracy_cost(
            x_target, x_sim, weights, precision_cost_matrix, interval
        )

        self.assertTrue(np.all(derivative_p_c[0, :, 0:3] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 4::] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 3] == 2 * (weights["w_p"] * x_target[0, :, 3])))

    def test_L2_cost(self):
        print(" Test L2 cost")
        dt = 0.1
        reference_result = 112.484456945 * dt
        weights = getdefaultweights()
        weights["w_2"] = 1.0
        u = self.get_arbitrary_array()
        L2_cost = cost_functions.control_strength_cost(u, weights, dt)
        self.assertAlmostEqual(L2_cost, reference_result, places=8)

    def test_derivative_L2_cost(self):
        print(" Test L2 cost derivative")
        u = self.get_arbitrary_array()
        desired_output = u
        self.assertTrue(np.all(cost_functions.derivative_L2_cost(u) == desired_output))

    def test_weights_dictionary(self):
        print("Test ditionary of cost weights")
        model = FHNModel()
        duration = 1.0
        model.params.duration = duration
        zero_input = ZeroInput().generate_input(duration=duration + model.params.dt, dt=model.params.dt)
        model.params["x_ext"] = zero_input
        model.params["y_ext"] = zero_input
        model.run()
        target = np.zeros((1, 2, 11))

        defaultweights = getdefaultweights()

        # use no reasonable input for weight dictionary
        for w in [None, 1, 0.1, dict()]:
            model_controlled = oc_fhn.OcFhn(model, target, weights=w)
            self.assertTrue(model_controlled.weights == defaultweights)
            model_controlled.optimize(0)  # check if dictionary is correctly implemented as numba dict

        # set only one parameter, others should be default
        for k in defaultweights.keys():
            w = dict()
            w[k] = 100.0
            model_controlled = oc_fhn.OcFhn(model, target, weights=w)
            self.assertTrue(model_controlled.weights[k] == w[k])
            for l in defaultweights.keys():
                if l == k:
                    continue
                self.assertTrue(model_controlled.weights[l] == defaultweights[l])

            model_controlled.optimize(0)  # check if dictionary is correctly implemented as numba dict


if __name__ == "__main__":
    unittest.main()

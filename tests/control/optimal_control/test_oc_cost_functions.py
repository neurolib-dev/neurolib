import unittest
import numpy as np
from neurolib.control.optimal_control import cost_functions
from neurolib.control.optimal_control.oc import getdefaultweights
from neurolib.models.fhn import FHNModel
from neurolib.control.optimal_control import oc_fhn

import test_oc_utils as test_oc_utils

p = test_oc_utils.params


class TestCostFunctions(unittest.TestCase):
    def test_precision_cost_full_timeseries(self):
        print(" Test precision cost full timeseries")
        N = 1
        precision_cost_matrix = np.ones((N, 2))
        dt = 0.1
        x_target = np.concatenate([p.TEST_INPUT_1N_6[:, np.newaxis, :], p.TEST_INPUT_1N_6[:, np.newaxis, :]], axis=1)
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
        target = np.concatenate(
            [p.TEST_INPUT_2N_6[:, np.newaxis, :], 2.0 * p.TEST_INPUT_2N_6[:, np.newaxis, :]], axis=1
        )
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
        target = np.concatenate([p.TEST_INPUT_1N_6[:, np.newaxis, :], p.TEST_INPUT_1N_6[:, np.newaxis, :]], axis=1)
        x_sim = np.copy(target)
        x_sim[0, :, 0] = -x_sim[0, :, 0]  # create setting where result depends only on this first entries
        interval = (0, target.shape[2])

        derivative_p_c = cost_functions.derivative_accuracy_cost(
            target, x_sim, weights, precision_cost_matrix, interval
        )

        self.assertTrue(np.all(derivative_p_c[0, :, 1::] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 0] == 2 * (weights["w_p"] * target[0, :, 0])))

    def test_derivative_precision_cost_full_timeseries_nodes_channels(self):
        print(" Test precision cost derivative full timeseries for node and channel selection")
        weights = getdefaultweights()
        N = 2
        target = np.concatenate(
            [p.TEST_INPUT_2N_6[:, np.newaxis, :], 2.0 * p.TEST_INPUT_2N_6[:, np.newaxis, :]], axis=1
        )
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
        target = np.concatenate([p.TEST_INPUT_1N_6[:, np.newaxis, :], p.TEST_INPUT_1N_6[:, np.newaxis, :]], axis=1)
        x_sim = np.copy(target)
        x_sim[0, :, 3] = -x_sim[0, :, 3]
        interval = (3, target.shape[2])
        weights = getdefaultweights()
        precision_cost = cost_functions.accuracy_cost(target, x_sim, weights, precision_cost_matrix, dt, interval)
        # Result should only depend on second half of the timeseries.
        self.assertEqual(precision_cost, weights["w_p"] / 2 * np.sum((2 * target[0, :, 3]) ** 2) * dt)

    def test_derivative_precision_cost_in_interval(self):
        """This test is analogous to the 'test_derivative_precision_cost'. However, the signal is repeated twice, but
        only the second interval is to be taken into account.
        """
        print(" Test precision cost derivative in time interval")
        weights = getdefaultweights()
        N = 1
        precision_cost_matrix = np.ones((N, 2))
        target = np.concatenate([p.TEST_INPUT_1N_6[:, np.newaxis, :], p.TEST_INPUT_1N_6[:, np.newaxis, :]], axis=1)
        x_sim = np.copy(target)
        x_sim[0, :, 3] = -x_sim[0, :, 3]  # create setting where result depends only on this first entries
        interval = (3, target.shape[2])
        derivative_p_c = cost_functions.derivative_accuracy_cost(
            target, x_sim, weights, precision_cost_matrix, interval
        )

        self.assertTrue(np.all(derivative_p_c[0, :, 0:3] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 4::] == 0))
        self.assertTrue(np.all(derivative_p_c[0, :, 3] == 2 * (weights["w_p"] * target[0, :, 3])))

    def test_L2_cost(self):
        print(" Test L2 cost")
        dt = 0.1
        reference_result = p.INT_INPUT_1N_6 * dt
        weights = getdefaultweights()
        weights["w_2"] = 1.0
        u = np.concatenate([p.TEST_INPUT_1N_6[:, np.newaxis, :], p.TEST_INPUT_1N_6[:, np.newaxis, :]], axis=1)
        L2_cost = cost_functions.control_strength_cost(u, weights, dt)
        self.assertAlmostEqual(L2_cost, reference_result, places=8)

    def test_derivative_L2_cost(self):
        print(" Test L2 cost derivative")
        u = np.concatenate([p.TEST_INPUT_1N_6[:, np.newaxis, :], p.TEST_INPUT_1N_6[:, np.newaxis, :]], axis=1)
        desired_output = u
        self.assertTrue(np.all(cost_functions.derivative_L2_cost(u) == desired_output))

    def test_L1D_cost(self):
        print(" Test L1D cost")
        dt = 0.1
        reference_result = 2.0 * np.sum(np.sqrt(np.sum(p.TEST_INPUT_1N_6**2 * dt, axis=1)))
        weights = getdefaultweights()
        weights["w_1D"] = 1.0
        u = np.concatenate([p.TEST_INPUT_1N_6[:, np.newaxis, :], p.TEST_INPUT_1N_6[:, np.newaxis, :]], axis=1)
        L1D_cost = cost_functions.control_strength_cost(u, weights, dt)

        self.assertAlmostEqual(L1D_cost, reference_result, places=8)

    def test_derivative_L1D_cost(self):
        print(" Test L1D cost derivative")
        dt = 0.1
        denominator = np.sqrt(np.sum(p.TEST_INPUT_1N_6**2 * dt, axis=1))

        u = np.concatenate([p.TEST_INPUT_1N_6[:, np.newaxis, :], p.TEST_INPUT_1N_6[:, np.newaxis, :]], axis=1)
        reference_result = np.zeros((u.shape))
        for n in range(u.shape[0]):
            for v in range(u.shape[1]):
                reference_result[n, v, :] = u[n, v, :] / denominator[n]

        self.assertTrue(np.all(cost_functions.derivative_L1D_cost(u, dt) == reference_result))

    def test_weights_dictionary(self):
        print("Test dictionary of cost weights")
        model = FHNModel()
        model.params.duration = p.TEST_DURATION_6
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

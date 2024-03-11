import unittest
import numpy as np
import numba
from neurolib.control.optimal_control.oc import (
    solve_adjoint,
    compute_gradient,
    update_control_with_limit,
    convert_interval,
    limit_control_to_interval,
)
from neurolib.control.optimal_control.oc_wc import OcWc
from neurolib.models.wc import WCModel
from neurolib.utils.stimulus import ZeroInput

import test_oc_utils as test_oc_utils

p = test_oc_utils.params


class TestOC(unittest.TestCase):
    """
    Test functions in neurolib/control/optimal_control/oc.py
    """

    @staticmethod
    def get_deterministic_wilson_cowan_test_setup():
        """Run a Wilson-Cowan model with default parameters for five time steps, use simulated time series to create
            a modified target. Set the inputs to the model to zero.
        :return: Instance of the Wilson-Cowan model with external inputs set to zero and duration of five time steps.
                 Target time series for exc- and inh- variable. The exc-variable is modified to be double the values
                 compared to obtained time series by simulating the model with default parameters.
        :rtype:  Tuple[neurolib.models.wc.model.WCModel, np.array]
        """
        model = WCModel()
        model.params["duration"] = p.TEST_DURATION_6
        model.run()

        target = test_oc_utils.gettarget_1n(model)
        target[0, 0, :] = target[0, 0, :] * 2.0
        test_oc_utils.set_input(model, p.ZERO_INPUT_1N_6)

        return model, target

    def test_solve_adjoint(self):
        print("Test solve adjoint method.")
        N, V, T = 1, 2, 6
        dt = 1.0

        jacobian = np.ones((N, T, V, V))
        jacobian_nw = np.zeros((N, N, T, V, V))
        fx = np.zeros((N, V, T))
        dh_dxdot = np.ones((N, V, V))
        vars_names = ["a", "b"]

        # # Fx = 0 should lead to adjoint = 0
        adjoint = solve_adjoint(
            [jacobian],
            [0],  # delay
            jacobian_nw,
            fx,
            (N, V, T),
            dt,
            N,
            T,
            np.zeros((N, N)).astype(int),
            dh_dxdot,
            vars_names,
            vars_names,
        )

        self.assertTrue(np.all(adjoint == np.zeros((adjoint.shape))))

        # Fx = 1 should lead to adjoint != 0
        fx = np.ones((N, V, T))

        adjoint = solve_adjoint(
            [jacobian],
            [0],  # delay
            jacobian_nw,
            fx,
            (N, V, T),
            dt,
            N,
            T,
            np.zeros((N, N)).astype(int),
            dh_dxdot,
            vars_names,
            vars_names,
        )

        result = np.zeros((T))
        result[-2] = -1.0
        result[-4] = -1.0
        result[-6] = -1.0

        self.assertTrue(np.all(adjoint == result))

    def test_compute_gradient(self):
        print("Test compute gradient method.")
        N, V, T = 1, 2, 6

        adjoint_state = np.zeros((N, V, T))
        control_matrix = np.ones((N, V))
        dh_du = np.ones((N, V, V, T))
        control_interval = numba.typed.List([0, T])
        df_du = np.zeros((N, V, T))

        # # df_du = 0 should lead to gadient = 0
        gradient = compute_gradient(
            N,
            V,
            V,
            df_du,
            adjoint_state,
            control_matrix,
            dh_du,
            control_interval,
        )

        self.assertTrue(np.all(gradient == np.zeros((gradient.shape))))

        df_du = np.ones((N, V, T))
        # df_du = 1 should lead to gadient == df_du
        gadient = compute_gradient(
            N,
            V,
            V,
            df_du,
            adjoint_state,
            control_matrix,
            dh_du,
            control_interval,
        )

        self.assertTrue(np.all(gadient == df_du))

        df_du = np.ones((N, V, T))
        adjoint_state = np.ones((N, V, T))
        # df_du = 1 and adjoint == 1 should lead to gadient != 0
        gadient = compute_gradient(
            N,
            V,
            V,
            df_du,
            adjoint_state,
            control_matrix,
            dh_du,
            control_interval,
        )

        result = 3.0 * np.ones((T))

        for n in range(N):
            self.assertTrue(np.all(gadient[n, :, :] == result))

    def test_update_input(self):
        # Run the test with an instance of an arbitrarily derived class
        # check update_input function
        print("Tets update_input function.")

        model, target = self.get_deterministic_wilson_cowan_test_setup()
        model_controlled = OcWc(model, target)
        model_controlled.control = np.concatenate(
            (p.TEST_INPUT_1N_6[:, np.newaxis, :], p.TEST_INPUT_1N_6[:, np.newaxis, :]), axis=1
        )

        for iv in model.input_vars:
            self.assertTrue((model_controlled.model.params[iv] == 0.0).all())

        model_controlled.update_input()

        for iv in model.input_vars:
            self.assertTrue((model_controlled.model.params[iv] == p.TEST_INPUT_1N_6).all())

    def test_step_size(self):
        # Run the test with an instance of an arbitrarily derived class.
        # This test case is not specific to any step size algorithm or initial step size.

        print("Test step size is larger zero.")

        model, target = self.get_deterministic_wilson_cowan_test_setup()

        cost_mat = np.zeros((model.params.N, len(model.output_vars)))
        control_mat = np.zeros((model.params.N, len(model.state_vars)))
        cost_mat[0, 0] = 1
        control_mat[0, 1] = 1

        model_controlled = OcWc(model, target, weights=None, cost_matrix=cost_mat, control_matrix=control_mat)

        self.assertTrue(model_controlled.step_size(-model_controlled.compute_gradient()) > 0.0)

    def test_step_size_no_step(self):
        # Run the test with an instance of an arbitrarily derived class.
        # Checks that for a zero-gradient no step is performed (i.e. step-size=0).

        print("Test step size is zero if gradient is zero.")

        model, target = self.get_deterministic_wilson_cowan_test_setup()

        cost_mat = np.zeros((model.params.N, len(model.output_vars)))
        control_mat = np.zeros((model.params.N, len(model.state_vars)))
        cost_mat[0, 0] = 1
        control_mat[0, 1] = 1

        model_controlled = OcWc(model, target, weights=None, cost_matrix=cost_mat, control_matrix=control_mat)

        self.assertTrue(model_controlled.step_size(-np.zeros(target.shape)) == 0.0)

    def test_update_control_with_limit_no_limit(self):
        # Test for the control to be limited.

        print("Test control update with and without strength limit.")

        control = np.concatenate((p.TEST_INPUT_2N_6[:, np.newaxis, :], p.TEST_INPUT_2N_6[:, np.newaxis, :]), axis=1)
        control[0, 0, -1] = np.inf
        control[0, 1 - 1] = -np.inf
        step = 1.0
        (N, dim_in, T) = control.shape
        cost_gradient = 4.0 * control

        u_max = None
        control_limited = update_control_with_limit(N, dim_in, T, control, step, cost_gradient, u_max)
        self.assertTrue(np.all(control_limited == control + step * cost_gradient))

        u_max = 0.1
        control_limited = update_control_with_limit(N, dim_in, T, control, step, cost_gradient, u_max)
        self.assertTrue(np.all(np.abs(control_limited) <= u_max))

    def test_limit_control_to_interval(self):
        # Test for the control to be unchanged, if no limit is set.

        print("Test limit of control interval.")

        control = np.concatenate((p.TEST_INPUT_2N_6[:, np.newaxis, :], p.TEST_INPUT_2N_6[:, np.newaxis, :]), axis=1)
        (N, dim_in, T) = control.shape

        c_int = (0, T)
        control_lim = limit_control_to_interval(N, dim_in, T, control, c_int)
        self.assertTrue(np.all(control_lim == control))

        c_int = (4, 6)
        control_lim = limit_control_to_interval(N, dim_in, T, control, c_int)
        self.assertTrue(np.all(control_lim[:, :, : c_int[0]]) == 0.0)
        self.assertTrue(np.all(control_lim[:, :, c_int[1] :]) == 0.0)
        self.assertTrue(np.all(control_lim[:, :, c_int[0] : c_int[1]] == control[:, :, c_int[0] : c_int[1]]))

    def test_convert_interval_none(self):
        print("Test convert interval.")
        array_length = 10  # arbitrary
        interval = (None, None)
        interval_converted = convert_interval(interval, array_length)
        self.assertTupleEqual(interval_converted, (0, array_length))

    def test_convert_interval_one_is_none(self):
        print("Test convert interval.")
        array_length = 10  # arbitrary
        interval = (0, None)
        interval_converted = convert_interval(interval, array_length)
        self.assertTupleEqual(interval_converted, (0, array_length))

    def test_convert_interval_negative(self):
        print("Test convert interval.")
        array_length = 10  # arbitrary
        interval = (-6, -2)
        interval_converted = convert_interval(interval, array_length)
        self.assertTupleEqual(interval_converted, (array_length + interval[0], array_length + interval[1]))

    def test_convert_interval_unchanged(self):
        print("Test convert interval.")
        array_length = 10  # arbitrary
        interval = (1, 7)  # arbitrary
        interval_converted = convert_interval(interval, array_length)
        self.assertTupleEqual(interval_converted, interval)

    def test_convert_interval_wrong_order(self):
        print("Test convert interval.")
        array_length = 10  # arbitrary
        interval = (5, -7)  # arbitrary
        self.assertRaises(AssertionError, convert_interval, interval, array_length)

    def test_convert_interval_invalid_range_negative(self):
        print("Test convert interval.")
        array_length = 10  # arbitrary
        interval = (-11, 5)  # arbitrary
        self.assertRaises(AssertionError, convert_interval, interval, array_length)

    def test_convert_interval_invalid_range_positive(self):
        print("Test convert interval.")
        array_length = 10  # arbitrary
        interval = (9, 11)  # arbitrary
        self.assertRaises(AssertionError, convert_interval, interval, array_length)


if __name__ == "__main__":
    unittest.main()

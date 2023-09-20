import unittest
import numpy as np
import neurolib
from neurolib.control.optimal_control.oc import (
    solve_adjoint,
    update_control_with_limit,
    convert_interval,
    limit_control_to_interval,
)
from neurolib.control.optimal_control.oc_wc import OcWc
from neurolib.models.wc import WCModel
from neurolib.utils.stimulus import ZeroInput


class TestOC(unittest.TestCase):
    """
    Test functions in neurolib/control/optimal_control/oc.py
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
            fall in the range [-5., 5.].
        :return: An array with arbitrary float and np.inf values.
        :rtype:  np.ndarray of shape 2 x 2 x 10
        """
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
        dt = model.params["dt"]
        duration = 5 * dt
        model.params["duration"] = duration
        model.run()

        target = np.concatenate(
            (
                np.concatenate((model.params["exc_init"], model.params["inh_init"]), axis=1)[:, :, np.newaxis],
                np.stack((model.exc, model.inh), axis=1),
            ),
            axis=2,
        )

        target[0, 0, :] = target[0, 0, :] * 2.0

        model.params["exc_ext"] = ZeroInput().generate_input(duration=duration + dt, dt=dt)
        model.params["inh_ext"] = ZeroInput().generate_input(duration=duration + dt, dt=dt)

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

    def test_step_size(self):
        # Run the test with an instance of an arbitrary derived class.
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
        # Run the test with an instance of an arbitrary derived class.
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

        control = self.get_arbitrary_array_finite_values()
        step = 1.0
        cost_gradient = self.get_arbitrary_array()
        (N, dim_in, T) = control.shape

        u_max = None
        control_limited = update_control_with_limit(N, dim_in, T, control, step, cost_gradient, u_max)
        self.assertTrue(np.all(control_limited == control + step * cost_gradient))

        u_max = 5.0
        control_limited = update_control_with_limit(N, dim_in, T, control, step, cost_gradient, u_max)
        self.assertTrue(np.all(np.abs(control_limited) <= u_max))

    def test_limit_control_to_interval(self):
        # Test for the control to be unchanged, if no limit is set.

        print("Test limit of control interval.")

        control = self.get_arbitrary_array_finite_values()
        (N, dim_in, T) = control.shape

        control_interval = (0, T)
        control_limited = limit_control_to_interval(N, dim_in, T, control, control_interval)
        self.assertTrue(np.all(control_limited == control))

        control_interval = (3, 7)
        control_limited = limit_control_to_interval(N, dim_in, T, control, control_interval)
        self.assertTrue(np.all(control_limited[:, :, : control_interval[0]]) == 0.0)
        self.assertTrue(np.all(control_limited[:, :, control_interval[1] :]) == 0.0)
        self.assertTrue(
            np.all(
                control_limited[:, :, control_interval[0] : control_interval[1]]
                == control[:, :, control_interval[0] : control_interval[1]]
            )
        )

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
        self.assertTupleEqual(interval_converted, (4, 8))

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

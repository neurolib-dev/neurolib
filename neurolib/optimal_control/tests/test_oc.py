import unittest
import numpy as np
from neurolib.optimal_control.oc import solve_adjoint, update_control_with_limit, convert_interval
from neurolib.optimal_control.oc_wc import OcWc
from neurolib.models.wc import WCModel
from neurolib.utils.stimulus import ZeroInput


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

    def test_step_size(self):
        # Run the test with an instance of an arbitrary derived class.
        # This test case is not specific to any step size algorithm or initial step size.
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

        prec_mat = np.zeros((model.params.N, len(model.output_vars)))
        control_mat = np.zeros((model.params.N, len(model.state_vars)))
        prec_mat[0, 0] = 1
        control_mat[0, 1] = 1
        model.params["exc_ext"] = ZeroInput().generate_input(duration=duration + dt, dt=dt)
        model.params["inh_ext"] = ZeroInput().generate_input(duration=duration + dt, dt=dt)

        model_controlled = OcWc(model, target, w_p=1, w_2=0, precision_matrix=prec_mat, control_matrix=control_mat)

        self.assertTrue(model_controlled.step_size(-model_controlled.compute_gradient()) > 0.0)

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

    def test_convert_interval_none(self):
        array_length = 10  # arbitrary
        interval = (None, None)
        interval_converted = convert_interval(interval, array_length)
        self.assertTupleEqual(interval_converted, (0, array_length))

    def test_convert_interval_one_is_none(self):
        array_length = 10  # arbitrary
        interval = (0, None)
        interval_converted = convert_interval(interval, array_length)
        self.assertTupleEqual(interval_converted, (0, array_length))

    def test_convert_interval_negative(self):
        array_length = 10  # arbitrary
        interval = (-6, -2)
        interval_converted = convert_interval(interval, array_length)
        self.assertTupleEqual(interval_converted, (4, 8))

    def test_convert_interval_unchanged(self):
        array_length = 10  # arbitrary
        interval = (1, 7)  # arbitrary
        interval_converted = convert_interval(interval, array_length)
        self.assertTupleEqual(interval_converted, interval)

    def test_convert_interval_wrong_order(self):
        array_length = 10  # arbitrary
        interval = (5, -7)  # arbitrary
        self.assertRaises(AssertionError, convert_interval, interval, array_length)

    def test_convert_interval_invalid_range_negative(self):
        array_length = 10  # arbitrary
        interval = (-11, 5)  # arbitrary
        self.assertRaises(AssertionError, convert_interval, interval, array_length)

    def test_convert_interval_invalid_range_positive(self):
        array_length = 10  # arbitrary
        interval = (9, 11)  # arbitrary
        self.assertRaises(AssertionError, convert_interval, interval, array_length)


if __name__ == "__main__":
    unittest.main()
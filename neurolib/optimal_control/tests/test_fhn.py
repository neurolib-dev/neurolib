import unittest

import numpy as np

from neurolib.models.fhn import FHNModel
from neurolib.utils.stimulus import ZeroInput
from neurolib.optimal_control import oc_fhn

global limit_diff
limit_diff = 1e-4


class TestFHN(unittest.TestCase):
    """
    Test fhn in neurolib/optimal_control/
    """

    def test_onenode_oc(self):
        fhn = FHNModel()

        duration = 3.
        a = 10.

        zero_input = ZeroInput().generate_input(duration=duration+fhn.params.dt, dt=fhn.params.dt)
        input_x = np.copy(zero_input)
        input_y = np.copy(input_x)
        for t in range(1, input_x.shape[1]-2):
            input_x[0, t] = np.random.uniform(-a, a)
        fhn.params["x_ext"] = input_x
        for t in range(1, input_y.shape[1]-2):
            input_y[0, t] = np.random.uniform(-a, a)
        fhn.params["y_ext"] = input_y

        fhn.params["duration"] = duration
        fhn.params["xs_init"] = np.array([[0.]])
        fhn.params["ys_init"] = np.array([[0.]])

        fhn.run()
        x_target = np.vstack([0., fhn.x.T])
        y_target = np.vstack([0., fhn.y.T])

        fhn.params["y_ext"] = zero_input
        fhn.params["x_ext"] = zero_input

        target = np.column_stack(( [x_target, y_target] )).T
        fhn_controlled = oc_fhn.OcFhn(fhn, target, w_p=1, w_2=0)

        control_coincide = False

        for i in range(100):
            fhn_controlled.optimize(1000)
            control = fhn_controlled.control

            c_diff = np.vstack( [np.abs(control[0,:] - input_x), np.abs(control[1,:] - input_y) ] )
            if np.amax(c_diff) < limit_diff:
                control_coincide = True
                break
        
        self.assertTrue(control_coincide)


if __name__ == "__main__":
    unittest.main()

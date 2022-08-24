from neurolib.optimal_control.oc import OC
import numba
import numpy as np


@numba.njit
def jacobian_hopf(a, w, V, x, y):
    """ Jacobian of systems dynamics for Hopf model.
        :param a:   Bifrucation parameter
        :type a :   float
        :param w:
        :type w:    float
        :param V:
        :type V:    int
        :param x:
        :type x:    float
        :param y:
        :type y:    float
    """
    jacobian = np.zeros((V, V))

    jacobian[0, :2] = [-a + 3 * x ** 2 + y ** 2, 2 * x * y + w]
    jacobian[1, :2] = [2 * x * y - w, -a + x ** 2 + 3 * y ** 2]

    return jacobian


class OcHopf(OC):
    # Remark: very similar to FHN!
    raise NotImplementedError


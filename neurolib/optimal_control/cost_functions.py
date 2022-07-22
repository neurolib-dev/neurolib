import numpy as np


def precision_cost(x_target: np.ndarray,
                   x_sim: np.ndarray,
                   w_p: float,
                   interval: tuple[int, int] = (0, None)) -> float:
    """ Summed squared difference between target and simulation within specified time interval weighted by w_p.

        :param x_target:    control-dimensions x T array that contains the target time series.

        :param x_sim:       control-dimensions x T array that contains the simulated time series.

        :param w_p:         Weight that is multiplied with the precision cost.

        :param interval:    [t_start, t_end]. Indices of start and end point of the slice (both inclusive) in time
                            dimension. Default is full time series.

        :return:            Precision cost for time interval.

    """
    # np.sum without specified axis implicitly performs
    # summation that would correspond to np.sum((x1(t)-x2(t)**2)
    # for the norm at one particular t as well as the integration over t
    # (commutative)
    return w_p * 0.5 * np.sum((x_target[:, interval[0]:interval[1]] - x_sim[:, interval[0]:interval[1]]) ** 2.)


def derivative_precision_cost(x_target: np.ndarray,
                              x_sim: np.ndarray,
                              w_p: float,
                              interval: tuple[int] = (0, None)) -> np.ndarray:
    """ Derivative of precision cost wrt. to x_sim.

        :param x_target:    control-dimensions x T array that contains the target time series.

        :param x_sim:       control-dimensions x T array that contains the simulated time series.

        :param w_p:         Weight that is multiplied with the precision cost.

        :param interval:    [t_start, t_end]. Indices of start and end point of the slice (both inclusive) in time
                            dimension. Default is full time series.

        :return:             control-dimensions x T array of precision cost gradients.
    """
    derivative = np.zeros(x_target.shape)
    derivative[:, interval[0]:interval[1]] = - w_p * (x_target[:, interval[0]:interval[1]]
                                                      - x_sim[:, interval[0]:interval[1]])
    return derivative


def energy_cost(u: np.ndarray, w_2: float) -> float:
    """
        :param u:   control-dimensions x T array. Control signals.

        :param w_2:  Weight that is multiplied with the W2 ("energy") cost.

        :return:    W2 cost of the control.
    """
    return w_2/2. * np.sum(u**2.)


def derivative_energy_cost(u: np.ndarray, w_2) -> np.ndarray:
    """
        :param u: control-dimensions x T array. Control signals.

        :param w_2: Weight that is multiplied with the W2 ("energy") cost.

        :return :   control-dimensions x T array of W2-cost gradients.
    """
    return w_2 * u

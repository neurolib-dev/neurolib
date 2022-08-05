import numpy as np


def precision_cost(x_target, x_sim, w_p, interval=(0, None)):
    """Summed squared difference between target and simulation within specified time interval weighted by w_p.

    :param x_target:    Control-dimensions x T array that contains the target time series.
    :type x_target:     np.ndarray

    :param x_sim:       Control-dimensions x T array that contains the simulated time series.
    :type x_sim:        np.ndarray

    :param w_p:         Weight that is multiplied with the precision cost.
    :type w_p:          float

    :param interval:    [t_start, t_end]. Indices of start and end point of the slice (both inclusive) in time
                        dimension. Default is full time series, defaults to (0, None).
    :type interval:     tuple, optional

    :return:            Precision cost for time interval.
    :rtype:             float

    """
    # np.sum without specified axis implicitly performs
    # summation that would correspond to np.sum((x1(t)-x2(t)**2)
    # for the norm at one particular t as well as the integration over t
    # (commutative)
    return (
        w_p
        * 0.5
        * np.sum(
            (
                x_target[:, interval[0] : interval[1]]
                - x_sim[:, interval[0] : interval[1]]
            )
            ** 2.0
        )
    )


def derivative_precision_cost(x_target, x_sim, w_p, interval=(0, None)):
    """Derivative of precision cost wrt. to x_sim.

    :param x_target:    Control-dimensions x T array that contains the target time series.
    :type x_target:     np.ndarray

    :param x_sim:       Control-dimensions x T array that contains the simulated time series.
    :type x_sim:        np.ndarray

    :param w_p:         Weight that is multiplied with the precision cost.
    :type w_p:          float

    :param interval:    [t_start, t_end]. Indices of start and end point of the slice (both inclusive) in time
                        dimension. Default is full time series, defaults to (0, None).
    :type interval:     tuple, optional

    :return:            Control-dimensions x T array of precision cost gradients.
    :rtype:             np.ndarray
    """
    derivative = np.zeros(x_target.shape)
    derivative[:, interval[0] : interval[1]] = -w_p * (
        x_target[:, interval[0] : interval[1]] - x_sim[:, interval[0] : interval[1]]
    )
    return derivative


def energy_cost(u, w_2):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_2: Weight that is multiplied with the W2 ("energy") cost.
    :type w_2:  float

    :return:    W2 cost of the control.
    :rtype:     float
    """
    return w_2 / 2.0 * np.sum(u**2.0)


def derivative_energy_cost(u, w_2):
    """
    :param u:   Control-dimensions x T array. Control signals.
    :type u:    np.ndarray

    :param w_2: Weight that is multiplied with the W2 ("energy") cost.
    :type w_2:  float

    :return :   Control-dimensions x T array of W2-cost gradients.
    :rtype:     np.ndarray
    """
    return w_2 * u

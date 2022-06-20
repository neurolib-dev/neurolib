import numpy as np


def adjustArrayShape(original, target):
    """
    Tiles and then cuts an array (or list or float) such that
    it has the same shape as target at the end.
    This is used to make sure that any input parameter like external current has
    the same shape as the rate array.
    """

    # make an ext_exc_current ARRAY from a LIST or INT
    if not hasattr(original, "__len__"):
        original = [original]
    original = np.array(original)

    # repeat original in y until larger (or same size) as target

    # tile until N

    # either (x,) shape or (y,x) shape
    if len(original.shape) == 1:
        # if original.shape[0] > 1:
        rep_y = target.shape[0]
    elif target.shape[0] > original.shape[0]:
        rep_y = int(target.shape[0] / original.shape[0]) + 1
    else:
        rep_y = 1

    # tile once so the array has shape (N,1)
    original = np.tile(original, (rep_y, 1))

    # tile until t

    if target.shape[1] > original.shape[1]:
        rep_x = int(target.shape[1] / original.shape[1]) + 1
    else:
        rep_x = 1
    original = np.tile(original, (1, rep_x))

    # cut from end because the beginning can be initial condition
    original = original[: target.shape[0], -target.shape[1] :]

    return original

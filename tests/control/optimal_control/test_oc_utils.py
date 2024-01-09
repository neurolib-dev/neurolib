import numpy as np
from neurolib.utils.collections import dotdict

params = dotdict({})

params.LIMIT_DIFF = 1e-4
params.LIMIT_DIFF_ID = 1e-12
params.TEST_DURATION_6 = 0.6
params.TEST_DURATION_8 = 0.8
params.TEST_DURATION_10 = 1.0
params.TEST_DURATION_12 = 1.2
params.TEST_DELAY = 0.2
params.ITERATIONS = 10000
params.LOOPS = 100

###################################################
ZERO_INPUT_1N_6 = np.zeros((1, 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))))
TEST_INPUT_1N_6 = ZERO_INPUT_1N_6.copy()
INIT_INPUT_1N_6 = ZERO_INPUT_1N_6.copy()

TEST_INPUT_1N_6[0, 1] = 2.0  # no negative values because rate inputs should be positive
TEST_INPUT_1N_6[0, 2] = 0.5
TEST_INPUT_1N_6[0, 3] = 1.0

INIT_INPUT_1N_6[0, 1] = TEST_INPUT_1N_6[0, 1] - 1e-2
INIT_INPUT_1N_6[0, 2] = TEST_INPUT_1N_6[0, 2] + 1e-3
INIT_INPUT_1N_6[0, 3] = TEST_INPUT_1N_6[0, 3] + 1e-2

params.ZERO_INPUT_1N_6 = ZERO_INPUT_1N_6
params.TEST_INPUT_1N_6 = TEST_INPUT_1N_6
params.INIT_INPUT_1N_6 = INIT_INPUT_1N_6

params.INT_INPUT_1N_6 = np.sum(TEST_INPUT_1N_6**2)

###################################################
ZERO_INPUT_1N_8 = np.zeros((1, 1 + int(np.around(params.TEST_DURATION_8 / 0.1, 1))))
TEST_INPUT_1N_8 = ZERO_INPUT_1N_8.copy()
INIT_INPUT_1N_8 = ZERO_INPUT_1N_8.copy()

TEST_INPUT_1N_8[
    :, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))
] = TEST_INPUT_1N_6
INIT_INPUT_1N_8[
    :, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))
] = INIT_INPUT_1N_6

params.ZERO_INPUT_1N_8 = ZERO_INPUT_1N_8
params.TEST_INPUT_1N_8 = TEST_INPUT_1N_8
params.INIT_INPUT_1N_8 = INIT_INPUT_1N_8

###################################################
ZERO_INPUT_1N_10 = np.zeros((1, 1 + int(np.around(params.TEST_DURATION_10 / 0.1, 1))))
TEST_INPUT_1N_10 = ZERO_INPUT_1N_10.copy()
INIT_INPUT_1N_10 = ZERO_INPUT_1N_10.copy()

TEST_INPUT_1N_10[
    :, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))
] = TEST_INPUT_1N_6
INIT_INPUT_1N_10[
    :, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))
] = INIT_INPUT_1N_6

params.ZERO_INPUT_1N_10 = ZERO_INPUT_1N_10
params.TEST_INPUT_1N_10 = TEST_INPUT_1N_10
params.INIT_INPUT_1N_10 = INIT_INPUT_1N_10

###################################################
ZERO_INPUT_1N_12 = np.zeros((1, 1 + int(np.around(params.TEST_DURATION_12 / 0.1, 1))))
TEST_INPUT_1N_12 = ZERO_INPUT_1N_12.copy()
INIT_INPUT_1N_12 = ZERO_INPUT_1N_12.copy()

TEST_INPUT_1N_12[
    :, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))
] = TEST_INPUT_1N_6
INIT_INPUT_1N_12[
    :, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))
] = INIT_INPUT_1N_6

params.ZERO_INPUT_1N_12 = ZERO_INPUT_1N_12
params.TEST_INPUT_1N_12 = TEST_INPUT_1N_12
params.INIT_INPUT_1N_12 = INIT_INPUT_1N_12

###################################################
ZERO_INPUT_2N_6 = np.zeros((2, 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))))
TEST_INPUT_2N_6 = ZERO_INPUT_2N_6.copy()
INIT_INPUT_2N_6 = ZERO_INPUT_2N_6.copy()

TEST_INPUT_2N_6[0, :] = TEST_INPUT_1N_6[0, :]
INIT_INPUT_2N_6[0, :] = INIT_INPUT_1N_6[0, :]

params.ZERO_INPUT_2N_6 = ZERO_INPUT_2N_6
params.TEST_INPUT_2N_6 = TEST_INPUT_2N_6
params.INIT_INPUT_2N_6 = INIT_INPUT_2N_6

###################################################
ZERO_INPUT_2N_8 = np.zeros((2, 1 + int(np.around(params.TEST_DURATION_8 / 0.1, 1))))
TEST_INPUT_2N_8 = ZERO_INPUT_2N_8.copy()
INIT_INPUT_2N_8 = ZERO_INPUT_2N_8.copy()

TEST_INPUT_2N_8[0, :] = TEST_INPUT_1N_8[0, :]
INIT_INPUT_2N_8[0, :] = INIT_INPUT_1N_8[0, :]

params.ZERO_INPUT_2N_8 = ZERO_INPUT_2N_8
params.TEST_INPUT_2N_8 = TEST_INPUT_2N_8
params.INIT_INPUT_2N_8 = INIT_INPUT_2N_8

###################################################
ZERO_INPUT_2N_10 = np.zeros((2, 1 + int(np.around(params.TEST_DURATION_10 / 0.1, 1))))
TEST_INPUT_2N_10 = ZERO_INPUT_2N_10.copy()
INIT_INPUT_2N_10 = ZERO_INPUT_2N_10.copy()

TEST_INPUT_2N_10[0, :] = TEST_INPUT_1N_10[0, :]
INIT_INPUT_2N_10[0, :] = INIT_INPUT_1N_10[0, :]

params.ZERO_INPUT_2N_10 = ZERO_INPUT_2N_10
params.TEST_INPUT_2N_10 = TEST_INPUT_2N_10
params.INIT_INPUT_2N_10 = INIT_INPUT_2N_10

###################################################
ZERO_INPUT_2N_12 = np.zeros((2, 1 + int(np.around(params.TEST_DURATION_12 / 0.1, 1))))
TEST_INPUT_2N_12 = ZERO_INPUT_2N_12.copy()
INIT_INPUT_2N_12 = ZERO_INPUT_2N_12.copy()

TEST_INPUT_2N_12[0, :] = TEST_INPUT_1N_12[0, :]
INIT_INPUT_2N_12[0, :] = INIT_INPUT_1N_12[0, :]

params.ZERO_INPUT_2N_12 = ZERO_INPUT_2N_12
params.TEST_INPUT_2N_12 = TEST_INPUT_2N_12
params.INIT_INPUT_2N_12 = INIT_INPUT_2N_12


def gettarget_1n(model):
    return np.concatenate(
        (
            np.concatenate(
                (model.params[model.init_vars[0]], model.params[model.init_vars[0]]),
                axis=1,
            )[:, :, np.newaxis],
            np.stack((model[model.state_vars[0]], model[model.state_vars[1]]), axis=1),
        ),
        axis=2,
    )


def gettarget_1n_ww(model):
    return np.concatenate(
        (
            np.concatenate(
                (
                    model.params[model.init_vars[0]],
                    model.params[model.init_vars[0]],
                    model.params[model.init_vars[2]],
                    model.params[model.init_vars[3]],
                ),
                axis=1,
            )[:, :, np.newaxis],
            np.stack(
                (
                    model[model.state_vars[0]],
                    model[model.state_vars[1]],
                    model[model.state_vars[2]],
                    model[model.state_vars[3]],
                ),
                axis=1,
            ),
        ),
        axis=2,
    )


def gettarget_2n(model):
    return np.concatenate(
        (
            np.stack(
                (
                    model.params[model.init_vars[0]][:, -1],
                    model.params[model.init_vars[1]][:, -1],
                ),
                axis=1,
            )[:, :, np.newaxis],
            np.stack((model[model.state_vars[0]], model[model.state_vars[1]]), axis=1),
        ),
        axis=2,
    )


def gettarget_2n_ww(model):
    return np.concatenate(
        (
            np.stack(
                (
                    model.params[model.init_vars[0]][:, -1],
                    model.params[model.init_vars[1]][:, -1],
                    model.params[model.init_vars[2]][:, -1],
                    model.params[model.init_vars[3]][:, -1],
                ),
                axis=1,
            )[:, :, np.newaxis],
            np.stack(
                (
                    model[model.state_vars[0]],
                    model[model.state_vars[1]],
                    model[model.state_vars[2]],
                    model[model.state_vars[3]],
                ),
                axis=1,
            ),
        ),
        axis=2,
    )


def setinitzero_1n(model):
    for init_var in model.init_vars:
        if "ou" in init_var:
            continue
        model.params[init_var] = np.array([[0.0]])


def setinitzero_2n(model):
    for init_var in model.init_vars:
        if "ou" in init_var:
            continue
        model.params[init_var] = np.zeros((2, 1))


def set_input(model, testinput):
    for iv in model.input_vars:
        model.params[iv] = testinput

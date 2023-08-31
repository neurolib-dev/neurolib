import numpy as np
from neurolib.utils.collections import dotdict

params = dotdict({})

params.LIMIT_DIFF = 1e-4
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

TEST_INPUT_1N_6[0, 1] = 1.0
TEST_INPUT_1N_6[0, 2] = -1.0
TEST_INPUT_1N_6[0, 3] = 0.5

INIT_INPUT_1N_6[0, 1] = TEST_INPUT_1N_6[0, 1] - 1e-2
INIT_INPUT_1N_6[0, 2] = TEST_INPUT_1N_6[0, 2] + 1e-3
INIT_INPUT_1N_6[0, 3] = TEST_INPUT_1N_6[0, 3] + 1e-2

params.ZERO_INPUT_1N_6 = ZERO_INPUT_1N_6
params.TEST_INPUT_1N_6 = TEST_INPUT_1N_6
params.INIT_INPUT_1N_6 = INIT_INPUT_1N_6

params.INT_INPUT_1N_6 = np.sum(TEST_INPUT_1N_6**2)
print(params.INT_INPUT_1N_6)


###################################################
ZERO_INPUT_1N_8 = np.zeros((1, 1 + int(np.around(params.TEST_DURATION_8 / 0.1, 1))))
TEST_INPUT_1N_8 = ZERO_INPUT_1N_8.copy()
INIT_INPUT_1N_8 = ZERO_INPUT_1N_8.copy()

TEST_INPUT_1N_8[:, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))] = TEST_INPUT_1N_6
INIT_INPUT_1N_8[:, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))] = INIT_INPUT_1N_6

params.ZERO_INPUT_1N_8 = ZERO_INPUT_1N_8
params.TEST_INPUT_1N_8 = TEST_INPUT_1N_8
params.INIT_INPUT_1N_8 = INIT_INPUT_1N_8

###################################################
ZERO_INPUT_1N_10 = np.zeros((1, 1 + int(np.around(params.TEST_DURATION_10 / 0.1, 1))))
TEST_INPUT_1N_10 = ZERO_INPUT_1N_10.copy()
INIT_INPUT_1N_10 = ZERO_INPUT_1N_10.copy()

TEST_INPUT_1N_10[:, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))] = TEST_INPUT_1N_6
INIT_INPUT_1N_10[:, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))] = INIT_INPUT_1N_6

params.ZERO_INPUT_1N_10 = ZERO_INPUT_1N_10
params.TEST_INPUT_1N_10 = TEST_INPUT_1N_10
params.INIT_INPUT_1N_10 = INIT_INPUT_1N_10

###################################################
ZERO_INPUT_1N_12 = np.zeros((1, 1 + int(np.around(params.TEST_DURATION_12 / 0.1, 1))))
TEST_INPUT_1N_12 = ZERO_INPUT_1N_12.copy()
INIT_INPUT_1N_12 = ZERO_INPUT_1N_12.copy()

TEST_INPUT_1N_12[:, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))] = TEST_INPUT_1N_6
INIT_INPUT_1N_12[:, : 1 + int(np.around(params.TEST_DURATION_6 / 0.1, 1))] = INIT_INPUT_1N_6

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
INIT_INPUT_2N_10 = ZERO_INPUT_1N_10.copy()

TEST_INPUT_2N_10[0, :] = TEST_INPUT_1N_10[0, :]
INIT_INPUT_2N_10[0, :] = INIT_INPUT_1N_10[0, :]

params.ZERO_INPUT_2N_10 = ZERO_INPUT_2N_10
params.TEST_INPUT_2N_10 = TEST_INPUT_2N_10
params.INIT_INPUT_2N_10 = INIT_INPUT_2N_10

###################################################
ZERO_INPUT_2N_12 = np.zeros((2, 1 + int(np.around(params.TEST_DURATION_12 / 0.1, 1))))
TEST_INPUT_2N_12 = ZERO_INPUT_2N_12.copy()
INIT_INPUT_2N_12 = ZERO_INPUT_1N_12.copy()

TEST_INPUT_2N_12[0, :] = TEST_INPUT_1N_12[0, :]
INIT_INPUT_2N_12[0, :] = INIT_INPUT_1N_12[0, :]

params.ZERO_INPUT_2N_12 = ZERO_INPUT_2N_12
params.TEST_INPUT_2N_12 = TEST_INPUT_2N_12
params.INIT_INPUT_2N_12 = INIT_INPUT_2N_12

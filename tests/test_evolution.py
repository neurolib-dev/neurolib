import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


import time
import unittest

import numpy as np

from neurolib.models.aln import ALNModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.evolution import Evolution

import neurolib.optimize.evolution.evolutionaryUtils as eu
import neurolib.utils.functions as func


class TestALNEvolution(unittest.TestCase):
    """
    Evolution with ALN model
    """

    def test_single_node(self):
        logging.info("\t > Evolution: Testing ALN single node ...")
        start = time.time()

        def evaluateSimulation(traj):
            rid = traj.id
            print("rin")
            logging.info("Running run id {}".format(rid))

            model = evolution.loadIndividual(traj)

            model.params["dt"] = 0.1
            model.params["duration"] = 2 * 1000.0

            model.run()

            # -------- fitness evaluation here --------

            # example: get dominant frequency of activity
            frs, powers = func.getPowerSpectrum(
                model.rates_exc[:, -int(1000 / model.params["dt"]) :],
                model.params["dt"],
            )
            domfr = frs[np.argmax(powers)]

            fitness = abs(domfr - 25)  # let's try to find a 25 Hz oscillation

            fitness_tuple = ()
            fitness_tuple += (fitness,)
            return fitness_tuple, model.outputs

        alnModel = ALNModel(simulateBOLD=True)
        alnModel.run()

        pars = ParameterSpace(
            ["mue_ext_mean", "mui_ext_mean"], [[0.0, 4.0], [0.0, 4.0]]
        )
        evolution = Evolution(
            alnModel,
            pars,
            evaluateSimulation,
            weightList=[-1.0],
            POP_INIT_SIZE=5,
            POP_SIZE=3,
            NGEN=3,
        )
        evolution.run(verbose=False)
        evolution.info(plot=False)
        traj = evolution.loadResults()
        gens, all_scores = evolution.getScoresDuringEvolution()

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


if __name__ == "__main__":
    unittest.main()

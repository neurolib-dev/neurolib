import logging
import time
import sys
import unittest
import pytest

import numpy as np

from neurolib.models.aln import ALNModel
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.evolution import Evolution

import neurolib.optimize.evolution.evolutionaryUtils as eu
import neurolib.utils.functions as func


class TestVanillaEvolution(unittest.TestCase):
    """Test of the evolutionary optimization without a neural model 
    """

    def test_circle_optimization(self):
        logging.info("\t > Evolution: Testing vanilla optimization of a circle ...")
        start = time.time()

        def optimize_me(traj):
            ind = evolution.getIndividualFromTraj(traj)
            computation_result = abs((ind.x ** 2 + ind.y ** 2) - 1)
            fitness_tuple = (computation_result,)
            result_dict = {"result": [computation_result]}
            return fitness_tuple, result_dict

        pars = ParameterSpace(["x", "y"], [[-5.0, 5.0], [-5.0, 5.0]])
        evolution = Evolution(optimize_me, pars, weightList=[-1.0], POP_INIT_SIZE=8, POP_SIZE=8, NGEN=2)
        evolution.run(verbose=False)

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


class TestALNEvolution(unittest.TestCase):
    """Evolution with ALN model
    """

    def test_single_node(self):
        logging.info("\t > Evolution: Testing ALN single node ...")
        start = time.time()

        def evaluateSimulation(traj):
            rid = traj.id
            logging.info("Running run id {}".format(rid))

            model = evolution.getModelFromTraj(traj)

            model.params["dt"] = 0.1
            model.params["duration"] = 2 * 1000.0

            model.run()

            # -------- fitness evaluation here --------

            # example: get dominant frequency of activity
            frs, powers = func.getPowerSpectrum(
                model.rates_exc[:, -int(1000 / model.params["dt"]) :], model.params["dt"],
            )
            domfr = frs[np.argmax(powers)]

            fitness = abs(domfr - 25)  # let's try to find a 25 Hz oscillation

            fitness_tuple = ()
            fitness_tuple += (fitness,)
            return fitness_tuple, model.outputs

        alnModel = ALNModel()
        alnModel.run(bold=True)

        pars = ParameterSpace(["mue_ext_mean", "mui_ext_mean"], [[0.0, 4.0], [0.0, 4.0]])
        evolution = Evolution(
            evaluateSimulation, pars, model=alnModel, weightList=[-1.0], POP_INIT_SIZE=6, POP_SIZE=4, NGEN=3,
        )
        evolution.run(verbose=False)
        evolution.info(plot=False)
        traj = evolution.loadResults()
        gens, all_scores = evolution.getScoresDuringEvolution()

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


if __name__ == "__main__":
    unittest.main()

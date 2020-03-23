import logging
import time
import unittest

import numpy as np

from neurolib.models.aln import ALNModel
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.parameterSpace import ParameterSpace
import neurolib.utils.functions as func


class TestALNExploration(unittest.TestCase):
    """
    ALN model parameter exploration with pypet.
    """

    def test_single_node(self):
        logging.info("\t > BoxSearch: Testing ALN single node ...")
        start = time.time()

        aln = ALNModel()
        parameters = ParameterSpace({"mue_ext_mean": np.linspace(0, 3, 2), "mui_ext_mean": np.linspace(0, 3, 2)})
        search = BoxSearch(aln, parameters)
        search.run()
        search.loadResults()

        for i in search.dfResults.index:
            search.dfResults.loc[i, "max_r"] = np.max(
                search.results[i]["rates_exc"][:, -int(1000 / aln.params["dt"]) :]
            )

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))

    def test_brain_network(self):
        from neurolib.utils.loadData import Dataset

        ds = Dataset("hcp")
        aln = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat, bold=True)
        # Resting state fits
        aln.params["mue_ext_mean"] = 1.57
        aln.params["mui_ext_mean"] = 1.6
        aln.params["sigma_ou"] = 0.09
        aln.params["b"] = 5.0
        aln.params["signalV"] = 2
        aln.params["dt"] = 0.2
        aln.params["duration"] = 0.2 * 60 * 1000

        # multi stage evaluation function
        def evaluateSimulation(traj):
            model = search.getModelFromTraj(traj)
            defaultDuration = model.params["duration"]
            invalid_result = {"fc": [0] * len(ds.BOLDs)}

            # -------- stage wise simulation --------

            # Stage 1 : simulate for a few seconds to see if there is any activity
            # ---------------------------------------
            model.params["dt"] = 0.1
            model.params["duration"] = 3 * 1000.0
            model.run()

            # check if stage 1 was successful
            if np.max(model.rates_exc[:, model.t > 500]) > 300 or np.max(model.rates_exc[:, model.t > 500]) < 10:
                search.saveOutputsToPypet(invalid_result, traj)
                return invalid_result, {}

            # Stage 2: simulate BOLD for a few seconds to see if it moves
            # ---------------------------------------
            model.params["dt"] = 0.2
            model.params["duration"] = 20 * 1000.0
            model.run(bold=True)

            if np.std(model.BOLD.BOLD[:, 5:10]) < 0.001:
                search.saveOutputsToPypet(invalid_result, traj)
                return invalid_result, {}

            # Stage 3: full and final simulation
            # ---------------------------------------
            model.params["dt"] = 0.2
            model.params["duration"] = defaultDuration
            model.run()

            # -------- evaluation here --------

            scores = []
            for i, fc in enumerate(ds.FCs):  # range(len(ds.FCs)):
                fc_score = func.matrix_correlation(func.fc(model.BOLD.BOLD[:, 5:]), fc)
                scores.append(fc_score)

            meanScore = np.mean(scores)
            result_dict = {"fc": meanScore}

            search.saveOutputsToPypet(result_dict, traj)

        # define and run exploration
        parameters = ParameterSpace({"mue_ext_mean": np.linspace(0, 3, 2), "mui_ext_mean": np.linspace(0, 3, 2)})
        search = BoxSearch(evalFunction=evaluateSimulation, model=aln, parameterSpace=parameters)
        search.run()


class TestCustomParameterExploration(unittest.TestCase):
    """Exploration with custom function
    """

    def test_circle_exploration(self):
        def explore_me(traj):
            pars = search.getParametersFromTraj(traj)
            # let's calculate the distance to a circle
            computation_result = abs((pars["x"] ** 2 + pars["y"] ** 2) - 1)
            result_dict = {"distance": computation_result}
            search.saveOutputsToPypet(result_dict, traj)

        parameters = ParameterSpace({"x": np.linspace(-2, 2, 2), "y": np.linspace(-2, 2, 2)})
        search = BoxSearch(evalFunction=explore_me, parameterSpace=parameters)
        search.run()
        search.loadResults(pypetShortNames=False)

        for i in search.dfResults.index:
            search.dfResults.loc[i, "distance"] = search.results[i]["distance"]

        search.dfResults


if __name__ == "__main__":
    unittest.main()

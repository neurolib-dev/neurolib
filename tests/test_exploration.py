import logging
import os
import random
import string
import time
import unittest

import neurolib.utils.paths as paths
import neurolib.utils.pypetUtils as pu
import numpy as np
import xarray as xr
from neurolib.models.aln import ALNModel
from neurolib.models.fhn import FHNModel
from neurolib.models.multimodel import MultiModel
from neurolib.models.multimodel.builder.fitzhugh_nagumo import FitzHughNagumoNetwork
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.loadData import Dataset
from neurolib.utils.parameterSpace import ParameterSpace


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))


class TestExplorationSingleNode(unittest.TestCase):
    """
    ALN single node exploration.
    """

    def test_single_node(self):
        start = time.time()

        model = ALNModel()
        parameters = ParameterSpace({"mue_ext_mean": np.linspace(0, 3, 2), "mui_ext_mean": np.linspace(0, 3, 2)})
        search = BoxSearch(model, parameters, filename="test_single_nodes.hdf")
        search.run()
        search.loadResults()
        dataarray = search.xr()
        self.assertTrue(isinstance(dataarray, xr.DataArray))
        self.assertFalse(dataarray.attrs)

        for i in search.dfResults.index:
            search.dfResults.loc[i, "max_r"] = np.max(
                search.results[i]["rates_exc"][:, -int(1000 / model.params["dt"]) :]
            )

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


class TestExplorationBrainNetwork(unittest.TestCase):
    """
    FHN brain network simulation with BOLD simulation.
    """

    def test_fhn_brain_network_exploration(self):
        ds = Dataset("hcp")
        model = FHNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        model.params.duration = 10 * 1000  # ms
        model.params.dt = 0.2
        model.params.bold = True
        parameters = ParameterSpace(
            {
                "x_ext": [np.ones((model.params["N"],)) * a for a in np.linspace(0, 2, 2)],
                "K_gl": np.linspace(0, 2, 2),
                "coupling": ["additive", "diffusive"],
            },
            kind="grid",
        )
        search = BoxSearch(model=model, parameterSpace=parameters, filename="test_fhn_brain_network_exploration.hdf")

        search.run(chunkwise=True, bold=True)

        pu.getTrajectorynamesInFile(os.path.join(paths.HDF_DIR, "test_fhn_brain_network_exploration.hdf"))
        search.loadDfResults()
        search.getRun(0, pypetShortNames=True)
        search.getRun(0, pypetShortNames=False)
        search.loadResults()
        # firing rate xr
        dataarray = search.xr()
        self.assertTrue(isinstance(dataarray, xr.DataArray))
        self.assertFalse(dataarray.attrs)
        # bold xr
        dataarray = search.xr(bold=True)
        self.assertTrue(isinstance(dataarray, xr.DataArray))
        self.assertFalse(dataarray.attrs)
        search.info()


class TestExplorationBrainNetworkPostprocessing(unittest.TestCase):
    """
    ALN brain network simulation with custom evaluation function.
    """

    @classmethod
    def setUpClass(cls):
        # def test_brain_network_postprocessing(self):
        ds = Dataset("hcp")
        model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        # Resting state fits
        model.params["mue_ext_mean"] = 1.57
        model.params["mui_ext_mean"] = 1.6
        model.params["sigma_ou"] = 0.09
        model.params["b"] = 5.0
        model.params["signalV"] = 2
        model.params["dt"] = 0.2
        model.params["duration"] = 0.2 * 60 * 1000

        # multi stage evaluation function
        def evaluateSimulation(traj):
            model = search.getModelFromTraj(traj)
            model.randomICs()
            model.params["dt"] = 0.2
            model.params["duration"] = 4 * 1000.0
            model.run(bold=True)

            result_dict = {"outputs": model.outputs}

            search.saveToPypet(result_dict, traj)

        # define and run exploration
        parameters = ParameterSpace({"mue_ext_mean": np.linspace(0, 3, 2), "mui_ext_mean": np.linspace(0, 3, 2)})
        search = BoxSearch(
            evalFunction=evaluateSimulation,
            model=model,
            parameterSpace=parameters,
            filename=f"test_brain_postprocessing_{randomString(20)}.hdf",
        )
        search.run()
        cls.model = model
        cls.search = search
        cls.ds = ds

    def test_getRun(self):
        self.search.getRun(0)

    def test_loadResults(self):
        self.search.loadResults()

    def test_loadResults_all_False(self):
        self.search.loadResults(all=False)


class TestCustomParameterExploration(unittest.TestCase):
    """Exploration with custom function"""

    def test_circle_exploration(self):
        def explore_me(traj):
            pars = search.getParametersFromTraj(traj)
            # let's calculate the distance to a circle
            computation_result = abs((pars["x"] ** 2 + pars["y"] ** 2) - 1)
            result_dict = {"distance": computation_result}
            search.saveToPypet(result_dict, traj)

        parameters = ParameterSpace({"x": np.linspace(-2, 2, 2), "y": np.linspace(-2, 2, 2)})
        search = BoxSearch(evalFunction=explore_me, parameterSpace=parameters, filename="test_circle_exploration.hdf")
        search.run()
        search.loadResults(pypetShortNames=False)

        for i in search.dfResults.index:
            search.dfResults.loc[i, "distance"] = search.results[i]["distance"]

        search.dfResults


class TestExplorationMultiModel(unittest.TestCase):
    """
    MultiModel exploration test - uses FHN network.
    """

    def test_multimodel_explore(self):
        start = time.time()

        DELAY = 13.0
        fhn_net = FitzHughNagumoNetwork(np.random.rand(2, 2), np.array([[0.0, DELAY], [DELAY, 0.0]]))
        model = MultiModel(fhn_net)
        parameters = ParameterSpace({"*noise*sigma": [0.0, 0.05], "*epsilon*": [0.5, 0.6]}, allow_star_notation=True)
        search = BoxSearch(model, parameters, filename="test_multimodel.hdf")
        search.run()
        search.loadResults()
        dataarray = search.xr()
        self.assertTrue(isinstance(dataarray, xr.DataArray))
        self.assertTrue(isinstance(dataarray.attrs, dict))
        self.assertListEqual(list(dataarray.attrs.keys()), list(parameters.dict().keys()))

        end = time.time()
        logging.info("\t > Done in {:.2f} s".format(end - start))


if __name__ == "__main__":
    unittest.main()

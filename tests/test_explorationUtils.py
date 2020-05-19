import unittest
import numpy as np
import pytest
import sys

from neurolib.models.fhn import FHNModel
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.utils.loadData import Dataset

import neurolib.optimize.exploration.explorationUtils as eu

import random
import string


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))


class TestExplorationUtils(unittest.TestCase):
    """
    Test functions in neurolib/optimize/exploration/explorationUtils.py
    """

    @classmethod
    def setUpClass(cls):
        ds = Dataset("hcp")
        model = FHNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        model.params.duration = 10 * 1000  # ms
        model.params.dt = 0.1
        model.params.bold = True
        parameters = ParameterSpace(
            {
                "x_ext": [np.ones((model.params["N"],)) * a for a in np.linspace(0, 2, 2)],
                "K_gl": np.linspace(0, 2, 2),
                "coupling": ["additive", "diffusive"],
            },
            kind="grid",
        )
        search = BoxSearch(
            model=model, parameterSpace=parameters, filename=f"test_exploration_utils_{randomString(20)}.hdf"
        )

        search.run(chunkwise=True, bold=True)

        search.loadResults()
        # flatten x_ext parameter
        search.dfResults.x_ext = [a[0] for a in list(search.dfResults.x_ext)]

        cls.model = model
        cls.search = search
        cls.ds = ds

    def test_processExplorationResults(self):
        eu.processExplorationResults(
            self.search, model=self.model, ds=self.ds, bold_transient=0
        )

    def test_findCloseResults(self):
        eu.findCloseResults(self.search.dfResults, dist=1, x_ext=0, K_gl=0.0)

    @pytest.mark.skipif(sys.platform in ["darwin", "win32"], reason="Testing plots does not work on macOS or Windows")
    def test_plotExplorationResults(self):
        eu.processExplorationResults(
            self.search, model=self.model, ds=self.ds, bold_transient=0
        )

        eu.plotExplorationResults(
            self.search.dfResults,
            par1=["x_ext", "$x_{ext}$"],
            par2=["K_gl", "$K$"],
            plot_key="max_" + self.model.default_output,
            contour="max_" + self.model.default_output,
            alpha="max_" + self.model.default_output,
            by=["coupling"],
            by_label=["coupling"],
            plot_key_label="testlabel",
            one_figure=True,
        )

        eu.plotExplorationResults(
            self.search.dfResults,
            par1=["x_ext", "$x_{ext}$"],
            par2=["K_gl", "$K$"],
            plot_key="max_" + self.model.default_output,
            by_label=["coupling"],
            plot_key_label="testlabel",
            one_figure=False,
        )

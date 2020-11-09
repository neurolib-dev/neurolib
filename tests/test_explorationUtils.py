import random
import string
import sys
import unittest

import neurolib.optimize.exploration.explorationUtils as eu
import numpy as np
import pytest
from neurolib.models.aln import ALNModel
from neurolib.optimize.exploration import BoxSearch
from neurolib.utils.loadData import Dataset
from neurolib.utils.parameterSpace import ParameterSpace


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
        model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        model.params.duration = 11 * 1000  # ms
        model.params.dt = 0.2
        parameters = ParameterSpace(
            {
                "mue_ext_mean": np.linspace(0, 3, 2),
                "mui_ext_mean": np.linspace(0, 3, 2),
                "b": [0.0, 10.0],
            },
            kind="grid",
        )
        search = BoxSearch(
            model=model, parameterSpace=parameters, filename=f"test_exploration_utils_{randomString(20)}.hdf"
        )

        search.run(chunkwise=True, bold=True)

        search.loadResults()

        cls.model = model
        cls.search = search
        cls.ds = ds

    def test_processExplorationResults(self):
        eu.processExplorationResults(self.search, model=self.model, ds=self.ds, bold_transient=0)

    def test_findCloseResults(self):
        eu.findCloseResults(self.search.dfResults, dist=1, mue_ext_mean=0, mui_ext_mean=0.0)

    @pytest.mark.skipif(sys.platform in ["darwin", "win32"], reason="Testing plots does not work on macOS or Windows")
    def test_plotExplorationResults(self):
        eu.processExplorationResults(self.search, model=self.model, ds=self.ds, bold_transient=0)

        # one_figure = True
        # alpha_mask
        # contour
        eu.plotExplorationResults(
            self.search.dfResults,
            par1=["mue_ext_mean", "$mue$"],
            par2=["mui_ext_mean", "$mui$"],
            plot_key="max_" + self.model.default_output,
            contour="max_" + self.model.default_output,
            alpha_mask="max_" + self.model.default_output,
            by=["b"],
            by_label=["b"],
            plot_key_label="testlabel",
            one_figure=True,
        )

        # one_figure = False
        eu.plotExplorationResults(
            self.search.dfResults,
            par1=["mue_ext_mean", "$mue$"],
            par2=["mui_ext_mean", "$mui$"],
            plot_key="max_" + self.model.default_output,
            by_label=["b"],
            plot_key_label="testlabel",
            one_figure=False,
        )

    @pytest.mark.skipif(sys.platform in ["darwin", "win32"], reason="Testing plots does not work on macOS or Windows")
    def test_plotRun(self):
        eu.plotResult(self.search, 0)


if __name__ == "__main__":
    unittest.main()

import logging
import time
import sys
import unittest
import pytest

import random
import numpy as np

from neurolib.utils.parameterSpace import ParameterSpace
from neurolib.optimize.evolution import Evolution

import neurolib.optimize.evolution.evolutionaryUtils as eu
import neurolib.optimize.evolution.deapUtils as du


class TestEvolutinUtils(unittest.TestCase):
    """
    Test functions in neurolib/utils/functions.py
    """

    @classmethod
    def setUpClass(cls):
        pars = ParameterSpace(
            ["mue_ext_mean", "mui_ext_mean", "b"],
            [[0.0, 3.0], [0.0, 3.0], [0.0, 100.0]],
        )
        evolution = Evolution(
            lambda v: v,
            pars,
            weightList=[1.0],
            POP_INIT_SIZE=4,
            POP_SIZE=4,
            NGEN=2,
            filename="TestEvolutinUtils.hdf",
        )

        cls.evolution = evolution

        # fake population
        pop = evolution.toolbox.population(n=100)
        fitness_length = 3

        for i, p in enumerate(pop):
            if random.random() < 0.1:
                fitnessesResult = [np.nan] * fitness_length
            else:
                fitnessesResult = np.random.random(fitness_length)
            p.id = i
            p.fitness.values = fitnessesResult
            p.fitness.score = np.nansum(p.fitness.wvalues) / (len(p.fitness.wvalues))
            p.gIdx = 0

        cls.pop = pop
        cls.evolution.pop = pop
        cls.evolution.gIdx = 1

    def test_getValidPopulation(self):
        self.evolution.getValidPopulation(self.pop)
        self.evolution.getInvalidPopulation(self.pop)

    def test_individualToDict(self):
        self.evolution.individualToDict(self.pop[0])

    def test_randomParametersAdaptive(self):
        du.randomParametersAdaptive(self.evolution.paramInterval)

    def test_mutateUntilValid(self):
        du.mutateUntilValid(
            self.pop, self.evolution.paramInterval, self.evolution.toolbox, maxTries=10
        )

    @pytest.mark.skipif(
        sys.platform == "darwin", reason="plotting does not work on macOS"
    )
    def test_plots(self):
        matplotlib = pytest.importorskip("matplotlib")
        eu.plotPopulation(self.evolution, plotScattermatrix=True)


class TestEvolutionCrossover(unittest.TestCase):
    def test_all_crossovers(self):
        def evo(traj):
            return (1,), {}

        pars = ParameterSpace(["x"], [[0.0, 4.0]])
        evolution = Evolution(
            evalFunction=evo, parameterSpace=pars, filename="TestEvolutionCrossover.hdf"
        )
        evolution.runInitial()
        init_pop = evolution.pop.copy()

        # perform crossover methods
        ind1, ind2 = init_pop[:2]
        du.cxNormDraw_adapt(ind1, ind2, 0.4)
        du.cxUniform_adapt(ind1, ind2, 0.4)
        du.cxUniform_normDraw_adapt(ind1, ind2, 0.4)

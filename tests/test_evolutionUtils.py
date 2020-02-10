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
        pars = ParameterSpace(["mue_ext_mean", "mui_ext_mean", "b"], [[0.0, 3.0], [0.0, 3.0], [0.0, 100.0]])
        evolution = Evolution(lambda v: v, pars, weightList=[1.0], POP_INIT_SIZE=4, POP_SIZE=4, NGEN=2)

        cls.evolution = evolution

        # fake population
        pop = evolution.toolbox.population(n=100)
        fitness_length = 3

        for i, p in enumerate(pop):
            if random.random() < 0.1:
                fitnessesResult = [np.nan] * fitness_length
            else:
                fitnessesResult = np.random.random(fitness_length)
            p.fitness.values = fitnessesResult
            p.fitness.score = np.nansum(p.fitness.wvalues) / (len(p.fitness.wvalues))
        cls.pop = pop

    def test_getValidPopulation(self):
        self.evolution.getValidPopulation(self.pop)
        self.evolution.getInvalidPopulation(self.pop)

    def test_individualToDict(self):
        self.evolution.individualToDict(self.pop[0])

    def test_randomParametersAdaptive(self):
        du.randomParametersAdaptive(self.evolution.paramInterval)

    def test_mutateUntilValid(self):
        du.mutateUntilValid(self.pop, self.evolution.paramInterval, self.evolution.toolbox, maxTries=10)

    @pytest.mark.skipif(sys.platform == "darwin", reason="plotting does not work on macOS")
    def test_plotPopulation(self):
        matplotlib = pytest.importorskip("matplotlib")
        eu.plotPopulation(self.pop, self.evolution.paramInterval, plotScattermatrix=True)


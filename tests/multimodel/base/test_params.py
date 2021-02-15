"""
Simple test for parameters.
"""

import unittest

import numpy as np
import sympy as sp
from neurolib.models.multimodel.builder.base.params import (
    count_float_params,
    float_params_to_individual_symbolic,
    float_params_to_vector_symbolic,
)


class TestSymbolicParams(unittest.TestCase):
    def test_count_float_params(self):
        DICT_IN = {"a": 40.0, "b": 15, "conn": np.random.rand(4, 4), "a.noise": 12.5}
        length = count_float_params(DICT_IN)
        self.assertEqual(length, 2)

    def test_float_params_to_vector_symbolic(self):
        DICT_IN = {"a": 40.0, "b": 15, "conn": np.random.rand(4, 4)}
        result = float_params_to_vector_symbolic(DICT_IN)
        self.assertEqual(len(result), len(DICT_IN))
        self.assertEqual(result.keys(), DICT_IN.keys())
        for k, v in result.items():
            self.assertTrue(isinstance(v, (sp.matrices.expressions.matexpr.MatrixElement, sp.MatrixSymbol)))
            if isinstance(v, sp.MatrixSymbol):
                self.assertTupleEqual(v.shape, DICT_IN[k].shape)

    def test_float_params_to_individual_symbolic(self):
        DICT_IN = {"a": 40.0, "b": 15, "conn": np.random.rand(4, 4)}
        result = float_params_to_individual_symbolic(DICT_IN)
        self.assertEqual(len(result), len(DICT_IN))
        self.assertEqual(result.keys(), DICT_IN.keys())
        for k, v in result.items():
            self.assertTrue(isinstance(v, (sp.Symbol, np.ndarray)))
            if isinstance(v, np.ndarray):
                self.assertTupleEqual(v.shape, DICT_IN[k].shape)
                self.assertTrue(all([isinstance(element, sp.Symbol) for element in v.flatten()]))


if __name__ == "__main__":
    unittest.main()

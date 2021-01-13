import unittest
import numpy as np
from neurolib.utils.parameterSpace import ParameterSpace


class TestParameterSpace(unittest.TestCase):
    def test_parameterspace_init(self):
        # init from list
        _ = ParameterSpace(["a", "b"], [[3], [3]])

        # init from dict
        _ = ParameterSpace({"a": [1, 2], "b": [1, 2]})

        # init from dict with numpy arrays
        _ = ParameterSpace({"a": np.zeros((3)), "b": np.ones((33))})

    def test_parameterspace_kind(self):
        # 'point'
        par = ParameterSpace(["a", "b"], [[10], [3.0]])
        self.assertEqual(par.kind, "point")

        # 'bound'
        par = ParameterSpace(["a", "b"], [[3.0, 5.0], [0.0, 3.0]])
        self.assertEqual(par.kind, "bound")

        # 'grid'
        par = ParameterSpace(["a", "b"], [[3.0, 3.5, 5.0], [0.0, 3.0]])
        self.assertEqual(par.kind, "grid")

    def test_parameterspace_attributes(self):
        par = ParameterSpace(["a", "b"], [[10, 8], [3.0]])
        par.a
        par["a"]
        par.b
        par["c"] = [1, 2, 3]
        par.lowerBound
        par.upperBound

    def test_conversions(self):
        par = ParameterSpace({"a": [1, 2], "b": [1, 2]})
        par.named_tuple_constructor
        par.named_tuple

        par.dict()


if __name__ == "__main__":
    unittest.main()

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
        parametrization = par.get_parametrization()
        self.assertTrue(all(len(lst) == 1 for lst in parametrization.values()))

        # 'bound'
        par = ParameterSpace(["a", "b"], [[3.0, 5.0], [0.0, 3.0]])
        self.assertEqual(par.kind, "bound")
        parametrization = par.get_parametrization()
        self.assertTrue(all(len(lst) == 2 for lst in parametrization.values()))

        # 'grid'
        par = ParameterSpace(["a", "b"], [[3.0, 3.5, 5.0], [0.0, 3.0]])
        self.assertEqual(par.kind, "grid")
        parametrization = par.get_parametrization()
        self.assertTrue(all(len(lst) == 6 for lst in parametrization.values()))

        # 'sequence'
        par = ParameterSpace(["a", "b"], [[3.0, 3.5, 5.0], [0.0, 3.0]], kind="sequence")
        self.assertEqual(par.kind, "sequence")
        parametrization = par.get_parametrization()
        self.assertTrue(all(len(lst) == 5 for lst in parametrization.values()))

        # `explicit`
        par = ParameterSpace(["a", "b"], [[3.0, 3.5, 12.5], [0.0, 3.0, 17.2]], kind="explicit")
        self.assertEqual(par.kind, "explicit")
        parametrization = par.get_parametrization()
        self.assertTrue(all(len(lst) == 3 for lst in parametrization.values()))

    def test_inflate_to_sequence(self):
        SHOULD_BE = {
            "a": [1, 2, None, None, None, None, None],
            "b": [None, None, 3, 4, 5, None, None],
            "c": [None, None, None, None, None, 12.0, 54.0],
        }
        par = ParameterSpace([], [])
        param_dict = {"a": [1, 2], "b": [3, 4, 5], "c": [12.0, 54.0]}

        result = par._inflate_to_sequence(param_dict)
        self.assertTrue(all(len(lst) == 7 for lst in result.values()))
        self.assertDictEqual(result, SHOULD_BE)

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

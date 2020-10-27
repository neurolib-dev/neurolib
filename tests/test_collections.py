import unittest

from neurolib.utils.collections import flat_dict_to_nested, flatten_nested_dict, star_dotdict


class TestCollections(unittest.TestCase):
    NESTED_DICT = {"a": {"b": "c", "d": "e"}}
    FLAT_DICT_DOT = {"a.b": "c", "a.d": "e"}
    PARAM_DICT = {"mass0": {"a": 0.4, "b": 1.2, "c": "float"}, "mass1": {"a": 0.4, "b": 1.2, "c": "int"}}
    PARAMS_ALL_A = {"mass0.a": 0.4, "mass1.a": 0.4}
    PARAMS_ALL_A_CHANGED = {"mass0.a": 0.7, "mass1.a": 0.7}

    def test_flatten_nested_dict(self):
        flat = flatten_nested_dict(self.NESTED_DICT, sep=".")
        self.assertDictEqual(flat, self.FLAT_DICT_DOT)

    def test_flat_unflat(self):
        flat = flatten_nested_dict(self.NESTED_DICT, sep=".")
        unflat = flat_dict_to_nested(flat)
        self.assertDictEqual(self.NESTED_DICT, unflat)

    def test_star_dotdict(self):
        params = star_dotdict(flatten_nested_dict(self.PARAM_DICT), sep=".")
        self.assertTrue(isinstance(params, star_dotdict))
        # try get params by star
        self.assertDictEqual(params["*a"], self.PARAMS_ALL_A)
        # change params by star
        params["*a"] = 0.7
        self.assertDictEqual(params["*a"], self.PARAMS_ALL_A_CHANGED)


if __name__ == "__main__":
    unittest.main()

import unittest

from neurolib.utils.collections import (
    BACKWARD_REPLACE,
    FORWARD_REPLACE,
    _sanitize_keys,
    flat_dict_to_nested,
    flatten_nested_dict,
    sanitize_dot_dict,
    star_dotdict,
    unwrap_star_dotdict,
)

from neurolib.models.multimodel.builder.wilson_cowan import WilsonCowanNode
from neurolib.models.multimodel import MultiModel


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
        # delete params
        del params["*a"]
        self.assertFalse("a" in params)

    def test_sanitize_keys(self):
        k = "mass1.tau*"
        k_san = _sanitize_keys(k, FORWARD_REPLACE)
        self.assertEqual(k_san, k.replace("*", "STAR"))
        k_back = _sanitize_keys(k_san, BACKWARD_REPLACE)
        self.assertEqual(k, k_back)

    def test_sanitize_dotdict(self):
        dct = {"mass1*tau": 2.5, "mass2*tau": 4.1, "mass2.x": 12.0}
        should_be = {"mass1STARtau": 2.5, "mass2STARtau": 4.1, "mass2.x": 12.0}
        dct_san = sanitize_dot_dict(dct, FORWARD_REPLACE)
        self.assertDictEqual(dct_san, should_be)
        dct_back = sanitize_dot_dict(dct_san, BACKWARD_REPLACE)
        self.assertDictEqual(dct_back, dct)

    def test_unwrap_star_dotdict(self):
        wc = MultiModel.init_node(WilsonCowanNode())
        dct = {"*tau": 2.5}
        should_be = {
            "WCnode_0.WCmassEXC_0.tau": 2.5,
            "WCnode_0.WCmassEXC_0.noise_0.tau": 2.5,
            "WCnode_0.WCmassINH_1.tau": 2.5,
            "WCnode_0.WCmassINH_1.noise_0.tau": 2.5,
        }
        unwrapped = unwrap_star_dotdict(dct, wc)
        self.assertDictEqual(unwrapped, should_be)

        dct = {"STARtau": 2.5}
        should_be = {
            "WCnode_0.WCmassEXC_0.tau": 2.5,
            "WCnode_0.WCmassEXC_0.noise_0.tau": 2.5,
            "WCnode_0.WCmassINH_1.tau": 2.5,
            "WCnode_0.WCmassINH_1.noise_0.tau": 2.5,
        }
        unwrapped = unwrap_star_dotdict(dct, wc, replaced_dict=BACKWARD_REPLACE)
        self.assertDictEqual(unwrapped, should_be)


if __name__ == "__main__":
    unittest.main()

"""
Tests of modelling class.
"""

import unittest

import symengine as se
from neurolib.models.multimodel.builder.base.neural_mass import NeuralMass
from neurolib.utils.stimulus import OrnsteinUhlenbeckProcess, ZeroInput


class MassTest(NeuralMass):
    required_params = ["a", "b"]
    num_state_variables = 1
    num_noise_variables = 2
    helper_variables = ["helper_test"]
    python_callbacks = ["test_callback"]
    _noise_input = [ZeroInput(), ZeroInput()]


class TestNeuralMass(unittest.TestCase):

    PARAMS = {"a": 1.2, "b": 11.9}

    def test_init(self):
        mass = MassTest(self.PARAMS)
        self.assertTrue(isinstance(mass, NeuralMass))
        self.assertTrue(isinstance(mass.__str__(), str))
        self.assertEqual(mass.__str__(), mass.__repr__())
        self.assertTrue(isinstance(mass.describe(), dict))
        mass._initialize_state_vector()
        self.assertEqual(len(mass.initial_state), mass.num_state_variables)
        self.assertTrue(hasattr(mass, "DESCRIPTION_FIELD"))
        self.assertTrue(all(hasattr(mass, field) for field in mass.DESCRIPTION_FIELD))
        self.assertTrue(hasattr(mass, "_derivatives"))
        self.assertTrue(hasattr(mass, "_validate_params"))
        self.assertTrue(hasattr(mass, "_validate_callbacks"))
        self.assertTrue(hasattr(mass, "_initialize_state_vector"))
        self.assertTrue(all(isinstance(symb, se.Symbol) for symb in mass.helper_symbols.values()))
        # callbacks are UndefFunction for now
        self.assertTrue(all(isinstance(callback, se.UndefFunction) for callback in mass.callback_functions.values()))

    def test_derivatives(self):
        mass = MassTest(self.PARAMS)
        self.assertRaises(NotImplementedError, mass._derivatives)

    def test_validate_params(self):
        mass = MassTest(self.PARAMS)
        self.assertDictEqual(self.PARAMS, mass.params)

    def test_update_params(self):
        UPDATE_WITH = {"a": 2.4, "noise_0": {"seed": 12}}

        mass = MassTest(self.PARAMS)
        mass.index = 0
        mass.init_mass(start_idx_for_noise=0)
        self.assertEqual(mass.params["a"], self.PARAMS["a"])
        mass.update_params(UPDATE_WITH)
        self.assertEqual(mass.params["a"], UPDATE_WITH["a"])
        self.assertEqual(mass.params["noise_0"]["seed"], UPDATE_WITH["noise_0"]["seed"])

    def test_init_mass(self):
        mass = MassTest(self.PARAMS)
        self.assertFalse(mass.initialised)
        mass.index = 0
        mass.init_mass(start_idx_for_noise=6)
        self.assertTrue(mass.initialised)
        self.assertListEqual(mass.initial_state, [0.0] * mass.num_state_variables)
        self.assertListEqual(mass.noise_input_idx, [6, 7])

    def test_set_noise_input(self):
        mass = MassTest(self.PARAMS)
        self.assertTrue(all(isinstance(noise, ZeroInput) for noise in mass.noise_input))
        mass.noise_input = [OrnsteinUhlenbeckProcess(0.0, 0.0, 1.0), ZeroInput()]
        self.assertTrue(isinstance(mass.noise_input[0], OrnsteinUhlenbeckProcess))
        self.assertTrue(isinstance(mass.noise_input[1], ZeroInput))

    def test_unwrap_state_vector(self):
        mass = MassTest(self.PARAMS)
        mass.idx_state_var = 0
        self.assertTrue(hasattr(mass, "_unwrap_state_vector"))
        state_vec = mass._unwrap_state_vector()
        self.assertTrue(isinstance(state_vec, list))
        self.assertEqual(len(state_vec), mass.num_state_variables)
        self.assertTrue(all(isinstance(vec, se.Function) for vec in state_vec))


if __name__ == "__main__":
    unittest.main()

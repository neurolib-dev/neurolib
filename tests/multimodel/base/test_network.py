"""
Test for network class.
"""

import unittest
from copy import deepcopy

import numba
import numpy as np
import pytest
import symengine as se
import sympy as sp
from neurolib.models.multimodel.builder.base.constants import EXC, INH
from neurolib.models.multimodel.builder.base.network import (
    Network,
    Node,
    SingleCouplingExcitatoryInhibitoryNode,
    _sanitize_matrix,
)
from neurolib.models.multimodel.builder.base.neural_mass import NeuralMass
from neurolib.utils.stimulus import OrnsteinUhlenbeckProcess, ZeroInput

PARAMS = {"a": 1.2, "b": 11.9}


class TestSanitizeMatrix(unittest.TestCase):
    def test_sanitize_matrix_int(self):
        mat = (np.random.rand(2, 2) * 100.0).astype(int)
        result = _sanitize_matrix(mat, (2, 2))
        self.assertTrue(result.dtype.kind == "f")
        np.testing.assert_equal(mat.astype(float), result)

    def test_sanitize_matrix_float(self):
        mat = np.random.rand(2, 2)
        result = _sanitize_matrix(mat, (2, 2))
        self.assertTrue(result.dtype.kind == "f")
        np.testing.assert_equal(mat, result)

    def test_sanitize_matrix_wrong(self):
        mat = (np.random.rand(2, 2) * 100.0).astype(int)
        with pytest.raises(AssertionError):
            result = _sanitize_matrix(mat, (3, 3))


class ExcMassTest(NeuralMass):
    label = EXC
    required_params = ["a", "b"]
    coupling_variables = {0: "coupling_EXC"}
    state_variable_names = ["q"]
    num_state_variables = 1
    num_noise_variables = 2
    mass_type = EXC
    _noise_input = [ZeroInput(), ZeroInput()]


class InhMassTest(NeuralMass):
    label = INH
    required_params = ["a", "b"]
    coupling_variables = {0: "coupling_INH"}
    state_variable_names = ["q"]
    num_state_variables = 1
    num_noise_variables = 2
    mass_type = INH
    _noise_input = [ZeroInput(), ZeroInput()]


class NodeTest(Node):
    label = "TestNode"
    default_output = f"q_{EXC}"
    sync_variables = ["sync_test"]


class SingleCouplingNodeTest(SingleCouplingExcitatoryInhibitoryNode):
    label = "test_node"
    default_output = f"q_{EXC}"


class TestNode(unittest.TestCase):
    def _create_node(self):
        mass1 = ExcMassTest(PARAMS)
        mass1.index = 0
        mass2 = InhMassTest(PARAMS)
        mass2.index = 1
        node = NodeTest([mass1, mass2])
        return node

    def test_init(self):
        node = self._create_node()
        self.assertTrue(isinstance(node, Node))
        self.assertEqual(len(node), 2)
        self.assertEqual(node.num_state_variables, 2)
        self.assertEqual(node.num_noise_variables, 4)
        self.assertTrue(isinstance(node.__str__(), str))
        self.assertEqual(node.__str__(), node.__repr__())
        self.assertTrue(isinstance(node.describe(), dict))
        self.assertEqual(node.max_delay, 0.0)
        self.assertTrue(hasattr(node, "_derivatives"))
        self.assertTrue(hasattr(node, "_sync"))
        self.assertTrue(hasattr(node, "_callbacks"))
        self.assertTrue(hasattr(node, "default_output"))
        self.assertTrue(isinstance(node.default_network_coupling, dict))
        self.assertTrue(isinstance(node.sync_variables, list))
        self.assertTrue(node.are_params_floats)
        self.assertTrue(node.float_params is None)

    def test_sync(self):
        node = self._create_node()
        self.assertRaises(NotImplementedError, node._sync)

    def test_update_params(self):
        UPDATE_WITH = {"a": 2.4}

        node = self._create_node()
        self.assertEqual(node[0].params["a"], PARAMS["a"])
        self.assertEqual(node[1].params["a"], PARAMS["a"])
        node.update_params({f"{EXC}_0": UPDATE_WITH, f"{INH}_1": UPDATE_WITH})
        self.assertEqual(node[0].params["a"], UPDATE_WITH["a"])
        self.assertEqual(node[1].params["a"], UPDATE_WITH["a"])

    def test_make_params_symbolic(self):
        node = self._create_node()
        node.index = 0
        node.idx_state_var = 0
        node.init_node(start_idx_for_noise=6)
        # save orig params
        orig_params = deepcopy(node.get_nested_params())
        self.assertTrue(node.are_params_floats)
        node.make_params_symbolic(vector=False)
        self.assertFalse(node.are_params_floats)
        self.assertDictEqual(orig_params, node.float_params)
        for k, v in node[0].params.items():
            if "input" in k:
                continue
            self.assertTrue(isinstance(v, sp.Symbol))

        for k, v in node[1].params.items():
            if "input" in k:
                continue
            self.assertTrue(isinstance(v, sp.Symbol))

        node.make_params_floats()
        self.assertTrue(node.are_params_floats)
        self.assertDictEqual(orig_params, node.get_nested_params())
        self.assertTrue(node.float_params is None)

    def test_noise_input(self):
        node = self._create_node()
        self.assertTrue(all(isinstance(noise, ZeroInput) for noise in node.noise_input))
        node.noise_input = [
            OrnsteinUhlenbeckProcess(0.0, 0.0, 1.0),
            ZeroInput(),
            OrnsteinUhlenbeckProcess(0.0, 0.0, 1.0),
            ZeroInput(),
        ]
        self.assertTrue(isinstance(node.noise_input[0], OrnsteinUhlenbeckProcess))
        self.assertTrue(isinstance(node.noise_input[2], OrnsteinUhlenbeckProcess))
        self.assertTrue(isinstance(node.noise_input[1], ZeroInput))
        self.assertTrue(isinstance(node.noise_input[3], ZeroInput))

    def test_strip_index(self):
        node = self._create_node()
        self.assertEqual(node._strip_index("test_symb_1_2"), "test_symb_1")

    def test_all_couplings(self):
        ALL_COUPLING = {0: "coupling_EXC", 1: "coupling_INH"}
        node = self._create_node()
        all_couplings = node.all_couplings()
        self.assertDictEqual(all_couplings, ALL_COUPLING)

    def test_initial_state(self):
        node = self._create_node()
        node.index = 0
        node.init_node(start_idx_for_noise=0)
        np.testing.assert_equal(np.zeros((node.num_state_variables)), node.initial_state)
        # 1D case
        new_init_state = np.random.rand(node.num_state_variables)
        node.initial_state = new_init_state
        np.testing.assert_equal(node.initial_state, new_init_state)
        # 2D case
        new_init_state = np.random.normal(size=(node.num_state_variables, 5))
        node.initial_state = new_init_state
        np.testing.assert_equal(node.initial_state, new_init_state)

    def test_init_node(self):
        node = self._create_node()
        node.index = 0
        self.assertFalse(node.initialised)
        node.init_node(start_idx_for_noise=6)
        self.assertTrue(node.initialised)
        self.assertTrue(isinstance(node.get_nested_params(), dict))
        self.assertEqual(len(node.sync_symbols), 1)
        self.assertTrue(all(isinstance(symb, se.Symbol) for symb in node.sync_symbols.values()))
        np.testing.assert_equal(np.zeros((node.num_state_variables)), node.initial_state)
        for mass in node:
            self.assertListEqual(mass.noise_input_idx, [6, 7])
        self.assertListEqual(node.state_variable_names, [[f"q_{EXC}", f"q_{INH}"]])


class TestSingleCouplingExcitatoryInhibitoryNode(unittest.TestCase):
    def _create_node(self):
        mass1 = ExcMassTest(PARAMS)
        mass1.index = 0
        mass2 = InhMassTest(PARAMS)
        mass2.index = 1
        node = SingleCouplingNodeTest(
            [mass1, mass2],
            local_connectivity=np.random.rand(2, 2),
            local_delays=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        return node

    def test_init(self):
        node = self._create_node()
        self.assertTrue(isinstance(node, SingleCouplingExcitatoryInhibitoryNode))
        self.assertEqual(len(node.sync_variables), 4)
        self.assertTrue(isinstance(node.__str__(), str))
        self.assertTrue(isinstance(node.describe(), dict))
        self.assertEqual(node.max_delay, 4.0)
        self.assertEqual(np.array([0]), node.excitatory_masses)
        self.assertEqual(np.array([1]), node.inhibitory_masses)

    def test_update_params(self):
        UPDATE_WITH = {"a": 2.4}
        UPDATE_CONNECTIVITY = np.random.rand(2, 2)
        UPDATE_DELAYS = np.abs(np.random.rand(2, 2))
        node = self._create_node()
        self.assertEqual(node[0].params["a"], PARAMS["a"])
        self.assertEqual(node[1].params["a"], PARAMS["a"])
        node.update_params(
            {
                f"{EXC}_0": UPDATE_WITH,
                f"{INH}_1": UPDATE_WITH,
                "local_connectivity": UPDATE_CONNECTIVITY,
                "local_delays": UPDATE_DELAYS,
            }
        )
        self.assertEqual(node[0].params["a"], UPDATE_WITH["a"])
        self.assertEqual(node[1].params["a"], UPDATE_WITH["a"])
        np.testing.assert_equal(UPDATE_CONNECTIVITY, node.connectivity)
        np.testing.assert_equal(UPDATE_DELAYS, node.delays)

    def test_make_params_symbolic(self):
        node = self._create_node()
        node.index = 0
        node.idx_state_var = 0
        node.init_node(start_idx_for_noise=6)
        # save orig params
        orig_params = deepcopy(node.get_nested_params())
        self.assertTrue(node.are_params_floats)
        node.make_params_symbolic(vector=False)
        self.assertFalse(node.are_params_floats)
        for k, v in node[0].params.items():
            if "input" in k:
                continue
            self.assertTrue(isinstance(v, sp.Symbol))

        for k, v in node[1].params.items():
            if "input" in k:
                continue
            self.assertTrue(isinstance(v, sp.Symbol))

        self.assertTrue(isinstance(node.connectivity, np.ndarray))
        self.assertTrue(all(isinstance(element, sp.Symbol) for element in node.connectivity.flatten()))
        self.assertTrue(isinstance(node.delays, np.ndarray))
        self.assertTrue(all(isinstance(element, sp.Symbol) for element in node.delays.flatten()))
        node.make_params_floats()
        self.assertTrue(node.are_params_floats)
        self.assertTrue(node.float_params is None)

    def test_init_node(self):
        node = self._create_node()
        node.index = 0
        node.idx_state_var = 0
        node.init_node()
        self.assertEqual(len(node.sync_symbols), 4)
        self.assertTrue(all(isinstance(symb, se.Symbol) for symb in node.sync_symbols.values()))
        np.testing.assert_equal(np.zeros((node.num_state_variables)), node.initial_state)
        self.assertEqual(node.inputs.shape[0], len(node))
        self.assertEqual(len(node._sync()), 4)
        for helper_sync in node._sync():
            self.assertTrue(isinstance(helper_sync, tuple))
            self.assertEqual(len(helper_sync), 2)
            self.assertTrue(isinstance(helper_sync[0], se.Symbol))
            self.assertTrue(isinstance(helper_sync[1], se.Mul))


class TestNetwork(unittest.TestCase):
    def _create_network(self):
        mass1 = ExcMassTest(PARAMS)
        mass1.index = 0
        mass2 = InhMassTest(PARAMS)
        mass2.index = 1
        node1 = SingleCouplingNodeTest(
            [mass1, mass2],
            local_connectivity=np.random.rand(2, 2),
            local_delays=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        node1.index = 0
        node1.idx_state_var = 0
        node2 = deepcopy(node1)
        node2.index = 1
        node2.idx_state_var = node1.num_state_variables
        net = Network(
            [node1, node2],
            np.random.rand(2, 2),
            None,
        )
        net.label = "test_net"
        net.sync_variables = ["test"]
        net.init_network()
        # define subs for testing values
        substitutions = {f"current_y({idx})": 1.0 for idx in range(net.num_state_variables)}

        return net, substitutions

    def test_init(self):
        net, _ = self._create_network()
        self.assertTrue(net.initialised)
        self.assertTrue(isinstance(net, Network))
        self.assertEqual(len(net), net.num_nodes)
        self.assertTrue(isinstance(net.__str__(), str))
        self.assertEqual(net.__str__(), net.__repr__())
        self.assertTrue(isinstance(net.describe(), dict))
        self.assertTrue(isinstance(net.get_nested_params(), dict))
        self.assertTrue(hasattr(net, "_callbacks"))
        self.assertEqual(net.max_delay, 4.0)
        self.assertEqual(len(net.sync_symbols), len(net.sync_variables) * net.num_nodes)
        self.assertEqual(net.default_output, net[0].default_output)
        self.assertEqual(net.default_output, net[1].default_output)
        self.assertTrue(net.are_params_floats)
        self.assertTrue(net.float_params is None)

    def test_update_params(self):
        UPDATE_CONNECTIVITY = np.random.rand(2, 2)
        UPDATE_DELAYS = np.abs(np.random.rand(2, 2))
        UPDATE_WITH = {"a": 2.4}
        net, _ = self._create_network()
        net.update_params(
            {"connectivity": UPDATE_CONNECTIVITY, "delays": UPDATE_DELAYS, "test_node_0": {f"{EXC}_0": UPDATE_WITH}}
        )
        np.testing.assert_equal(net.connectivity, UPDATE_CONNECTIVITY)
        np.testing.assert_equal(net.delays, UPDATE_DELAYS)
        self.assertEqual(net[0][0].params["a"], UPDATE_WITH["a"])

    def test_set_noise_input(self):
        net, _ = self._create_network()
        self.assertTrue(all(isinstance(noise, ZeroInput) for noise in net.noise_input))
        net.noise_input = [OrnsteinUhlenbeckProcess(0.0, 0.0, 1.0)] * net.num_noise_variables
        self.assertTrue(all(isinstance(noise, OrnsteinUhlenbeckProcess) for noise in net.noise_input))

    def test_initial_state(self):
        net, _ = self._create_network()
        np.testing.assert_equal(np.zeros((net.num_state_variables)), net.initial_state)
        # 1D case
        new_init_state = np.random.rand(net.num_state_variables)
        net.initial_state = new_init_state
        np.testing.assert_equal(net.initial_state, new_init_state)
        # 2D case
        new_init_state = np.random.normal(size=(net.num_state_variables, 5))
        net.initial_state = new_init_state
        np.testing.assert_equal(net.initial_state, new_init_state)

    def test_prepare_mass_params(self):
        net, _ = self._create_network()
        # dict
        dict_params = net._prepare_mass_params({"a": 3}, num_nodes=len(net), native_type=dict)
        self.assertListEqual(dict_params, [{"a": 3}, {"a": 3}])
        # arrays
        array = np.random.rand(4, 4)
        array_params = net._prepare_mass_params(array, num_nodes=len(net), native_type=np.ndarray)
        self.assertListEqual(array_params, [array] * len(net))

    def test_strip_index(self):
        net, _ = self._create_network()
        self.assertEqual(net._strip_index("test_symb_1_2"), "test_symb_1")
        self.assertEqual(net._strip_node_idx("test_symb_1_2"), 2)

    def test_input_mat(self):
        net, _ = self._create_network()
        input_mat = net._construct_input_matrix(0)
        self.assertTupleEqual(input_mat.shape, (net.num_nodes, net.num_nodes))
        self.assertTrue(all(isinstance(coupling, se.Function) for coupling in input_mat.flatten()))

    def test_input_mat_with_None(self):
        net, _ = self._create_network()
        input_mat = net._construct_input_matrix([0, None])
        self.assertTupleEqual(input_mat.shape, (net.num_nodes, net.num_nodes))
        # assert from first node it's not 0, it's a Function
        self.assertTrue(all(isinstance(coupling, se.Function) for coupling in input_mat[:, 0]))
        # assert from second node it's 0
        self.assertTrue(all(coupling == 0.0 for coupling in input_mat[:, 1]))

    def test_no_coupling(self):
        net, _ = self._create_network()
        no_coupling = net._no_coupling(net.sync_variables[0])
        self.assertEqual(len(no_coupling), net.num_nodes)
        for coupling in no_coupling:
            self.assertTrue(isinstance(coupling, tuple))
            self.assertEqual(len(coupling), 2)
            self.assertTrue(isinstance(coupling[0], se.Symbol))
            self.assertEqual(coupling[1], 0.0)

    def test_diffusive_coupling(self):
        net, subs = self._create_network()
        diff_coupling = net._diffusive_coupling(0, net.sync_variables[0])
        self.assertEqual(len(diff_coupling), net.num_nodes)
        for coupling in diff_coupling:
            self.assertTrue(isinstance(coupling, tuple))
            self.assertEqual(len(coupling), 2)
            self.assertTrue(isinstance(coupling[0], se.Symbol))
            self.assertTrue(isinstance(coupling[1], se.Mul))
            evaluated = se.sympify(coupling[1]).subs(subs)
            self.assertEqual(float(evaluated), 0.0)

    def test_multiplicative_coupling(self):
        net, subs = self._create_network()
        multip_coupling = net._multiplicative_coupling(0, net.sync_variables[0])
        self.assertEqual(len(multip_coupling), net.num_nodes)
        for i, coupling in enumerate(multip_coupling):
            self.assertTrue(isinstance(coupling, tuple))
            self.assertEqual(len(coupling), 2)
            self.assertTrue(isinstance(coupling[0], se.Symbol))
            self.assertTrue(isinstance(coupling[1], se.Add))
            evaluated = se.sympify(coupling[1]).subs(subs)
            self.assertEqual(float(evaluated), net.connectivity.sum(axis=1)[i])

    def test_additive_coupling(self):
        net, subs = self._create_network()
        add_coupling = net._additive_coupling(0, net.sync_variables[0])
        self.assertEqual(len(add_coupling), net.num_nodes)
        for i, coupling in enumerate(add_coupling):
            self.assertTrue(isinstance(coupling, tuple))
            self.assertEqual(len(coupling), 2)
            self.assertTrue(isinstance(coupling[0], se.Symbol))
            self.assertTrue(isinstance(coupling[1], se.Add))
            evaluated = se.sympify(coupling[1]).subs(subs)
            np.testing.assert_allclose(float(evaluated), net.connectivity.sum(axis=1)[i])

    def test_additive_coupling_multiplier(self):
        MULTIPLIER = 2.4
        net, subs = self._create_network()
        add_coupling = net._additive_coupling(0, net.sync_variables[0], connectivity=MULTIPLIER * net.connectivity)

        self.assertEqual(len(add_coupling), net.num_nodes)
        for i, coupling in enumerate(add_coupling):
            self.assertTrue(isinstance(coupling, tuple))
            self.assertEqual(len(coupling), 2)
            self.assertTrue(isinstance(coupling[0], se.Symbol))
            self.assertTrue(isinstance(coupling[1], se.Add))
            evaluated = se.sympify(coupling[1]).subs(subs)
            np.testing.assert_allclose(
                float(evaluated),
                MULTIPLIER * net.connectivity.sum(axis=1)[i],
            )

    def test_sync(self):
        net, _ = self._create_network()
        self.assertListEqual(net._sync(), net[0]._sync() + net[1]._sync())


if __name__ == "__main__":
    unittest.main()

"""
Test for network class.
"""

import unittest
from copy import deepcopy

import numpy as np
import symengine as se
from neurolib.models.multimodel.builder.base.constants import EXC, INH
from neurolib.models.multimodel.builder.base.network import Network, Node, SingleCouplingExcitatoryInhibitoryNode
from neurolib.models.multimodel.builder.base.neural_mass import NeuralMass

PARAMS = {"a": 1.2, "b": 11.9}


class ExcMassTest(NeuralMass):
    label = EXC
    required_parameters = ["a", "b"]
    coupling_variables = {0: "coupling_EXC"}
    state_variable_names = ["q"]
    num_state_variables = 1
    num_noise_variables = 2
    mass_type = EXC


class InhMassTest(NeuralMass):
    label = INH
    required_parameters = ["a", "b"]
    coupling_variables = {0: "coupling_INH"}
    state_variable_names = ["q"]
    num_state_variables = 1
    num_noise_variables = 2
    mass_type = INH


class NodeTest(Node):
    default_output = f"q_{EXC}"
    sync_variables = ["sync_test"]


class SingleCouplingNodeTest(SingleCouplingExcitatoryInhibitoryNode):
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
        self.assertTrue(isinstance(node.describe(), dict))
        self.assertEqual(node.max_delay, 0.0)
        self.assertTrue(hasattr(node, "_derivatives"))
        self.assertTrue(hasattr(node, "_sync"))
        self.assertTrue(hasattr(node, "_callbacks"))
        self.assertTrue(hasattr(node, "default_output"))
        self.assertTrue(isinstance(node.default_network_coupling, dict))
        self.assertTrue(isinstance(node.sync_variables, list))

    def test_update_parameters(self):
        UPDATE_WITH = {"a": 2.4}

        node = self._create_node()
        node.update_parameters({"mass_0": UPDATE_WITH, "mass_1": UPDATE_WITH})
        self.assertDictEqual({**PARAMS, **UPDATE_WITH}, node[0].parameters)
        self.assertDictEqual({**PARAMS, **UPDATE_WITH}, node[1].parameters)

    def test_strip_index(self):
        node = self._create_node()
        self.assertEqual(node._strip_index("test_symb_1_2"), "test_symb_1")

    def test_all_couplings(self):
        ALL_COUPLING = {0: "coupling_EXC", 1: "coupling_INH"}
        node = self._create_node()
        all_couplings = node.all_couplings()
        self.assertDictEqual(all_couplings, ALL_COUPLING)

    def test_init_node(self):
        node = self._create_node()
        node.index = 0
        self.assertFalse(node.initialised)
        node.init_node(start_idx_for_noise=6)
        self.assertTrue(node.initialised)
        self.assertTrue(isinstance(node.get_nested_parameters(), dict))
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
            [mass1, mass2], local_connectivity=np.random.rand(2, 2), local_delays=np.array([[1.0, 2.0], [3.0, 4.0]]),
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

    def test_update_parameters(self):
        UPDATE_WITH = {"a": 2.4}
        UPDATE_CONNECTIVITY = np.random.rand(2, 2)
        node = self._create_node()
        node.update_parameters(
            {"mass_0": UPDATE_WITH, "mass_1": UPDATE_WITH, "local_connectivity": UPDATE_CONNECTIVITY}
        )
        self.assertDictEqual({**PARAMS, **UPDATE_WITH}, node[0].parameters)
        self.assertDictEqual({**PARAMS, **UPDATE_WITH}, node[1].parameters)
        np.testing.assert_equal(UPDATE_CONNECTIVITY, node.connectivity)

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
            [mass1, mass2], local_connectivity=np.random.rand(2, 2), local_delays=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        node1.index = 0
        node1.idx_state_var = 0
        node2 = deepcopy(node1)
        node2.index = 1
        node2.idx_state_var = node1.num_state_variables
        net = Network([node1, node2], np.random.rand(2, 2), None,)
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
        self.assertTrue(isinstance(net.describe(), dict))
        self.assertTrue(isinstance(net.get_nested_parameters(), dict))
        self.assertTrue(hasattr(net, "_callbacks"))
        self.assertEqual(net.max_delay, 4.0)
        self.assertEqual(len(net.sync_symbols), len(net.sync_variables) * net.num_nodes)
        self.assertEqual(net.default_output, net[0].default_output)
        self.assertEqual(net.default_output, net[1].default_output)

    def test_update_parameters(self):
        UPDATE_CONNECTIVITY = np.random.rand(2, 2)
        UPDATE_DELAYS = np.abs(np.random.rand(2, 2))
        net, _ = self._create_network()
        net.update_parameters({"connectivity": UPDATE_CONNECTIVITY, "delays": UPDATE_DELAYS})
        np.testing.assert_equal(net.connectivity, UPDATE_CONNECTIVITY)
        np.testing.assert_equal(net.delays, UPDATE_DELAYS)

    def test_prepare_mass_parameters(self):
        net, _ = self._create_network()
        # dict
        dict_params = net._prepare_mass_parameters({"a": 3}, num_nodes=len(net), native_type=dict)
        self.assertListEqual(dict_params, [{"a": 3}, {"a": 3}])
        # arrays
        array = np.random.rand(4, 4)
        array_params = net._prepare_mass_parameters(array, num_nodes=len(net), native_type=np.ndarray)
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
            np.testing.assert_allclose(float(evaluated), net.connectivity.sum(axis=0)[i])

    def test_additive_coupling_multiplier(self):
        MULTIPLIER = 2.4
        net, subs = self._create_network()
        add_coupling = net._additive_coupling(0, net.sync_variables[0], connectivity_multiplier=MULTIPLIER)

        self.assertEqual(len(add_coupling), net.num_nodes)
        for i, coupling in enumerate(add_coupling):
            self.assertTrue(isinstance(coupling, tuple))
            self.assertEqual(len(coupling), 2)
            self.assertTrue(isinstance(coupling[0], se.Symbol))
            self.assertTrue(isinstance(coupling[1], se.Add))
            evaluated = se.sympify(coupling[1]).subs(subs)
            np.testing.assert_allclose(
                float(evaluated), MULTIPLIER * net.connectivity.sum(axis=0)[i],
            )

    def test_sync(self):
        net, _ = self._create_network()
        self.assertListEqual(net._sync(), net[0]._sync() + net[1]._sync())


if __name__ == "__main__":
    unittest.main()

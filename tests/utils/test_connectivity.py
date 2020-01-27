"""
Set of basic tests for connectivity, i.e. connectome matrices.

(c) neurolib-devs
"""

import os
import unittest

import numpy as np

import networkx as nx
from neurolib.utils.connectivity import (
    BaseConnectivity,
    ConnectivityEnsemble,
    DelayMatrix,
    FiberLengthConnectivity,
    FunctionalConnectivity,
    StructuralConnectivity,
)

TEST_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "test_results"
)


class TestBaseConnectivity(unittest.TestCase):
    """
    Basic tests for connectivity matrices.
    """

    dummy_matrix = np.array([[1.32, 0.7], [-0.3, 1.3]])
    symm_matrix = np.array([[1.32, 0.2], [0.2, 1.3]])
    norm_matrix = np.array([[1.29411765, 0.49009803], [-0.21004201, 0.65]])
    adj_density_matrix = np.array([[1.32, 0.0], [-0.0, 0.0]])

    def test_init(self):
        mat = BaseConnectivity(self.dummy_matrix)
        self.assertEqual(mat.no_rois, 2)
        self.assertEqual(mat.shape, (2, 2))
        np.testing.assert_equal(mat.matrix, self.dummy_matrix)
        np.testing.assert_equal(mat.matrix[mat.triu_indices], np.array([0.7]))
        np.testing.assert_equal(
            mat.matrix[mat.triu_indices_w_diagonal], np.array([1.32, 0.7, 1.3])
        )

    def test_load_from_file(self):
        # set connectivity
        mat = BaseConnectivity(self.dummy_matrix)
        # load from mat file
        from_file = BaseConnectivity.from_mat_file(
            os.path.join(TEST_FOLDER, "dummy.mat"), key="mat_test"
        )
        np.testing.assert_equal(mat.matrix, from_file.matrix)
        # load from h5 file
        from_file = BaseConnectivity.from_hdf_file(
            os.path.join(TEST_FOLDER, "dummy.h5"), key="h5_test"
        )
        np.testing.assert_equal(mat.matrix, from_file.matrix)

    def test_single_node(self):
        weight = 8
        mat = BaseConnectivity.dummy_single_node(weight)
        np.testing.assert_equal(mat.matrix, np.atleast_2d(weight))

    def test_basic_functions(self):
        mat = BaseConnectivity(self.dummy_matrix)
        # assert basic functions
        np.testing.assert_almost_equal(
            mat.symmetrize_matrix(mat.matrix), self.symm_matrix
        )
        np.testing.assert_almost_equal(
            mat.normalize_matrix(mat.matrix), self.norm_matrix
        )

    def test_nx_graph(self):
        # test graph creation
        mat = BaseConnectivity(self.dummy_matrix, symmetrize=True)
        mat.create_nx_graph()
        graph = nx.Graph(mat.matrix)
        self.assertTrue(isinstance(mat.graph, nx.Graph))
        # compare graphs using isomorphism
        em = nx.algorithms.isomorphism.numerical_edge_match("weight", 1)
        self.assertTrue(nx.is_isomorphic(graph, mat.graph, edge_match=em))

    def test_adjust_density(self):
        mat = BaseConnectivity(self.dummy_matrix)
        mat.adjust_density(target_density=0.5)
        np.testing.assert_equal(mat.matrix, self.adj_density_matrix)


class TestStructuralConnectivity(unittest.TestCase):
    """
    Tests for structural connectivity matrix.
    """

    dummy_matrix = np.array([[1.32, 0.7], [-0.3, 1.3]])
    scaled_matrix = np.array([[1.0, 0.53030303], [-0.22727273, 0.98484848]])

    def test_scale_to_relative(self):
        mat = StructuralConnectivity(self.dummy_matrix)
        mat.scale_to_relative()
        np.testing.assert_almost_equal(mat.matrix, self.scaled_matrix)


class TestFunctionalConnectivity(unittest.TestCase):
    """
    Tests for functional connectivity matrix.
    """

    means = [0.5, -0.5, 1.4]
    cov_mat = np.array([[1.2, -0.7, 1.6], [-0.7, 0.5, 2.3], [1.6, 2.3, 0.83]])

    def test_init_from_timeseries(self):
        np.random.seed(42)
        timeseries = np.random.multivariate_normal(
            self.means, self.cov_mat, size=(10000)
        )
        # covariance FC
        fc_cov = FunctionalConnectivity.from_timeseries(
            timeseries, time_as_row=True, fc_type="cov"
        )
        # compute covariance as XX^T
        x_tilde = timeseries - np.mean(timeseries, axis=0)
        cov_computed = x_tilde.T.dot(x_tilde) / x_tilde.shape[0]
        np.testing.assert_almost_equal(fc_cov.matrix, cov_computed, decimal=3)
        # correlation FC
        fc_corr = FunctionalConnectivity.from_timeseries(
            timeseries, time_as_row=True, fc_type="corr"
        )
        # compute correlation from cov
        diag = np.diag(np.sqrt(np.diag(cov_computed)))  # sqrt(diagonal) matrix
        corr_computed = np.dot(
            np.dot(np.linalg.inv(diag), cov_computed), np.linalg.inv(diag)
        )
        np.testing.assert_almost_equal(fc_corr.matrix, corr_computed)


class TestFiberLengthConnectivity(unittest.TestCase):
    """
    Tests for fiber lengths matrix.
    """

    fiber_mat = np.array([[0.0, 3.2, 12.3], [3.2, 0.0, 0.9], [12.3, 0.9, 0.0]])
    sig_velocity = 7.5  # m/s

    def test_transform_to_delay_matrix(self):
        fib_conn = FiberLengthConnectivity(self.fiber_mat)
        # signal velocity is 7.5m/s with each segment being 1mm long
        delay_mat = fib_conn.transform_to_delay_matrix(
            signal_velocity=self.sig_velocity, segment_length=1.0
        )
        self.assertEqual(delay_mat.connectivity_type, "delay")
        np.testing.assert_equal(
            delay_mat.matrix, self.fiber_mat / self.sig_velocity
        )


class TestDelayMatrix(unittest.TestCase):
    """
    Tests for delay matrix.
    """

    fiber_mat = np.array([[0.0, 3.2, 12.3], [3.2, 0.0, 0.9], [12.3, 0.9, 0.0]])
    sig_velocity = 7.5  # m/s
    diagonal_delay = 0.06  # ms
    dt = 1e-6  # s

    def test_max_delay(self):
        delay_mat = DelayMatrix(self.fiber_mat / self.sig_velocity)
        # check maximal delay
        self.assertTrue(
            delay_mat.max_delay, np.max(self.fiber_mat / self.sig_velocity)
        )

    def test_unit_delay_diagonal(self):
        delay_mat = DelayMatrix(self.fiber_mat / self.sig_velocity)
        delay_mat.unit_delay_diagonal(self.diagonal_delay)
        # check delay matrix diagonal
        np.testing.assert_equal(
            np.diag(delay_mat.matrix),
            np.ones(self.fiber_mat.shape[0]) * self.diagonal_delay,
        )

    def test_transform_to_multiples_dt(self):
        delay_mat = DelayMatrix(self.fiber_mat / self.sig_velocity)
        delay_mat.transform_to_multiples_dt(dt=self.dt, dt_in_seconds=True)
        np.testing.assert_equal(
            delay_mat.matrix,
            np.around(
                (self.fiber_mat / self.sig_velocity) / (1000 * self.dt)
            ).astype(np.int),
        )


class TestConnectivityEnsamble(unittest.TestCase):
    """
    Basic tests for connectivity ensemble.
    """

    def test_load_from_folder(self):
        for file_type, key in zip(["mat", "h5"], ["mat_test", "h5_test"]):
            conn_ensemble = ConnectivityEnsemble.from_folder(
                TEST_FOLDER,
                file_type=file_type,
                connectivity_type="structural",
                key=key,
            )
            self.assertEqual(conn_ensemble.no_rois, 2)
            self.assertEqual(conn_ensemble.no_subjects, 1)
            self.assertTrue(
                all(
                    isinstance(conn, StructuralConnectivity)
                    for conn in conn_ensemble.original_matrices
                )
            )

    def test_aggregated_conns(self):
        conn_ensemble = ConnectivityEnsemble.from_folder(
            TEST_FOLDER,
            file_type="mat",
            connectivity_type="structural",
            key="mat_test",
        )
        # since we have only one subject, the mean is equal to that one subject
        np.testing.assert_equal(
            conn_ensemble.mean_connectivity,
            conn_ensemble.original_matrices[0].matrix,
        )
        np.testing.assert_equal(
            conn_ensemble.median_connectivity,
            conn_ensemble.original_matrices[0].matrix,
        )

    def test_rebuild_grand_matrix(self):
        conn_ensemble = ConnectivityEnsemble.from_folder(
            TEST_FOLDER,
            file_type="mat",
            connectivity_type="structural",
            key="mat_test",
        )
        self.assertEqual(conn_ensemble.no_subjects, 1)
        # create new matrix by added 0.5 to existing one
        new_matrix = conn_ensemble.original_matrices[0].matrix.copy() + 0.5
        # create structural connectivity out of it
        new_conn = StructuralConnectivity(new_matrix)
        # append to ensemble
        conn_ensemble.original_matrices.append(new_conn)
        # rebuild grand
        conn_ensemble.rebuild_grand_matrix()
        # check we have more subjects
        self.assertEqual(conn_ensemble.no_subjects, 2)
        # check mean and median
        manual_conn = np.dstack(
            [conn_ensemble.original_matrices[0].matrix, new_matrix]
        ).transpose(2, 0, 1)
        np.testing.assert_equal(
            conn_ensemble.mean_connectivity, np.mean(manual_conn, axis=0)
        )
        np.testing.assert_equal(
            conn_ensemble.median_connectivity, np.median(manual_conn, axis=0)
        )


if __name__ == "__main__":
    unittest.main()

"""
Set of tools for brain connectivities.

(c) neurolib-devs

# TODO do we do this upper file part?
"""

import logging
from glob import glob

import numpy as np

import networkx as nx
from h5py import File
from scipy.io import loadmat

from .atlases import BaseAtlas  # TODO relative or absolute imports?

MATLAB_EXT = ".mat"
HDF_EXT = ".h5"


class BaseConnectivity:
    """
    Represents basic connectivity within the brain according to some atlas.
    Implements basic methods.
    """

    connectivity_type = None
    label = ""

    @classmethod
    def dummy_single_node(cls, self_weight=1.0):
        """
        Init class as dummy single node connectivity.
        """

        return cls(np.ones((1, 1)) * self_weight, False, False, False, None)

    @classmethod
    def from_mat_file(
        cls,
        filename,
        key=None,
        nullify_diagonal=False,
        symmetrize=False,
        normalize=False,
        atlas=None,
    ):
        """
        Init class from matlab `.mat` file.
        """
        loaded_data = loadmat(filename)
        if key is None:
            key = [
                k for k in list(loaded_data.keys()) if not k.startswith("__")
            ][0]
        matrix = np.array(loaded_data[key])
        return cls(matrix, nullify_diagonal, symmetrize, normalize, atlas)

    @classmethod
    def from_hdf_file(
        cls,
        filename,
        key=None,
        nullify_diagonal=False,
        symmetrize=False,
        normalize=False,
        atlas=None,
    ):
        """
        Init class from hdf `.h5` file.
        """
        loaded_data = File(filename, "r")
        if key is None:
            key = next(iter(loaded_data.keys()))
        matrix = np.array(loaded_data[key])
        loaded_data.close()
        return cls(matrix, nullify_diagonal, symmetrize, normalize, atlas)

    def __init__(
        self,
        matrix,
        nullify_diagonal=False,
        symmetrize=False,
        normalize=False,
        atlas=None,
    ):
        """
        Init class with optional preprocessing.

        :param matrix: connectivity matrix - should be 2D
        :type matrix: np.ndarray
        :param nullify_diagonal: whether to insert zeros at the diagonal, i.e.
            at self-connections
        :type nullify_diagonal: bool
        :param symmetrize: whether to symmetrize the matrix
        :type symmetrize: bool
        :param normalize: whether to normalize the matrix to unit row and
            column sum
        :type normalize: bool
        :param atlas: brain parcellation atlas, if known
        :type atlas: str|tools.atlases.BaseAtlas|None
        """
        assert matrix.ndim == 2, "Connectivity must be 2D"
        assert (
            matrix.shape[0] == matrix.shape[1]
        ), "Non-square matrices not supported"
        self.matrix = matrix.astype(np.float)
        # optional preprocess
        if nullify_diagonal:
            np.fill_diagonal(self.matrix, 0.0)
        if symmetrize:
            self.matrix = self.symmetrize_matrix(self.matrix)
        if normalize:
            self.matrix = self.normalize_matrix(self.matrix)

        self.graph = None
        # if BaseAtlas, check number of nodes
        if isinstance(atlas, BaseAtlas):
            assert self.no_rois == atlas.no_rois
            self.label += f" ({atlas.label})"
        elif isinstance(atlas, str):
            self.label += f" ({atlas})"
        self.atlas = atlas

    def __str__(self):
        return (
            f"{self.label} matrix with {self.no_rois} ROIs of "
            f"{self.connectivity_type} type."
        )

    @property
    def no_rois(self):
        """
        Return number of ROIs.
        """
        return self.matrix.shape[0]

    @property
    def shape(self):
        """
        Return full shape of the matrix.
        """
        return self.matrix.shape

    @property
    def triu_indices(self):
        """
        Return upper triangle indices from connectivity matrix.
        """
        return np.triu_indices(self.no_rois, k=1)

    @property
    def triu_indices_w_diagonal(self):
        """
        Return upper triangle indices including the diagonal from connectivity
        matrix.
        """
        return np.triu_indices(self.no_rois, k=0)

    @staticmethod
    def symmetrize_matrix(matrix):
        """
        Symmetrize single square matrix.
        """
        return (matrix + matrix.transpose()) / 2.0

    @staticmethod
    def normalize_matrix(matrix):
        """
        Normalize single square matrix.
        """
        d = np.diag(np.power(np.sum(matrix, axis=0), -0.5))
        return np.dot(np.dot(d, matrix), d)

    def adjust_density(self, target_density):
        """
        Adjust density of matrix such that it has target density.

        :param target_density: desired density
        :type target_density: float
        """
        assert (
            0 < target_density < 1
        ), f"Density must be 0-1; got {target_density}"
        # number of needed nonzero elements
        nonzero_elements = self.no_rois * (self.no_rois - 1) * target_density
        # get threshold element
        threshold = np.sort(self.matrix.flatten())[::-1][int(nonzero_elements)]
        # threshold the matrix in-place
        self.matrix *= self.matrix > threshold

    def create_nx_graph(self, threshold=0.0, directed=False):
        """
        Create a networkx DiGraph representation from matrix.

        :param threshold: whether to threshold adjacency matrix
        :type threshold: float
        :param directed: whether to get directed or undirected graph
        :type directed: bool
        """
        graph = nx.DiGraph(self.matrix * (self.matrix > threshold))
        if directed:
            self.graph = graph
        else:
            self.graph = graph.to_undirected()


class ConnectivityEnsemble:
    """
    Represents an ensemble of connectivites, typically from more subjects.
    """

    @classmethod
    def from_folder(
        cls,
        path,
        file_type,
        connectivity_type,
        key=None,
        nullify_diagonal=False,
        symmetrize=False,
        normalize=False,
        atlas=None,
    ):
        """
        Load connectivity ensemble of given type from folder.

        :param path: path to connectivities
        :type path: str
        :param file_type: type of files from which to load connectivities:
            "mat" or "h5"
        :param connectivity_type: type of connectivities
        :type connectivity_type: str
        """
        if connectivity_type == "structural":
            base_type = StructuralConnectivity
        elif connectivity_type == "functional":
            base_type = FunctionalConnectivity
        elif connectivity_type == "fiber_lengths":
            base_type = FiberLengthConnectivity
        elif connectivity_type == "delay":
            base_type = DelayMatrix
        else:
            raise ValueError(f"Unknown connectivity type: {connectivity_type}")

        if file_type == "mat":
            ext = MATLAB_EXT
            loading_function = base_type.from_mat_file
        elif file_type == "h5":
            ext = HDF_EXT
            loading_function = base_type.from_hdf_file

        connectivities = []
        for conn_file in glob(f"{path}/*{ext}"):
            connectivities.append(
                loading_function(
                    conn_file,
                    key=key,
                    nullify_diagonal=nullify_diagonal,
                    symmetrize=symmetrize,
                    normalize=normalize,
                    atlas=atlas,
                )
            )

        return cls(connectivities)

    def __init__(self, matrices):
        """
        :param matrices: list of connectivites per subject
        :type matrices: list
        """
        assert all(isinstance(matrix, BaseConnectivity) for matrix in matrices)
        shapes = set(matrix.shape for matrix in matrices)
        assert len(shapes) == 1, f"Different shapes found: {shapes}"
        self.grand_matrix = np.dstack(
            [matrix.matrix for matrix in matrices]
        ).transpose(2, 0, 1)
        # save original matrices as a list for convenience
        self.original_matrices = matrices
        atlases = set(matrix.atlas for matrix in matrices)
        if len(atlases) == 1:
            self.atlas = matrices[0].atlas
        else:
            logging.warning(
                f"Multiple atlases found: {atlases}, not sure what to do..."
            )
            self.atlas = None

    def rebuild_grand_matrix(self):
        """
        Rebuild grand matrix, e.g. when doing some operation on individual
        matrices (stored in `original_matrices`) - any operation like this must
        be done inplace on the list with orig. matrices.
        """
        self.grand_matrix = np.dstack(
            [matrix.matrix for matrix in self.original_matrices]
        ).transpose(2, 0, 1)

    @property
    def no_rois(self):
        """
        Return number of ROIs.
        """
        return self.grand_matrix.shape[1]

    @property
    def no_subjects(self):
        """
        Return number of subject.
        """
        return self.grand_matrix.shape[0]

    @property
    def mean_connectivity(self):
        """
        Return mean connectivity matrix.
        """
        return np.nanmean(self.grand_matrix, axis=0)

    @property
    def median_connectivity(self):
        """
        Return median connectivity matrix.
        """
        return np.nanmedian(self.grand_matrix, axis=0)


class StructuralConnectivity(BaseConnectivity):
    """
    Structural connectivity matrix, usually from DTI measurement.
    """

    connectivity_type = "structural"
    label = "SC"

    def scale_to_relative(self):
        """
        Scale SC matrix to relative strengths between 0 and 1.
        """
        self.matrix /= np.max(self.matrix)


class FunctionalConnectivity(BaseConnectivity):
    """
    Functional connectivity matrix.
    """

    connectivity_type = "functional"
    label = "FC"

    @classmethod
    def from_timeseries(
        cls, timeseries, fc_type="corr", time_as_row=True, atlas=None
    ):
        """
        Init functional connectivity from timeseries.

        :param timeseries: timeseries for FC
        :fc_type timeseries: np.ndarray
        :param fc_type: type of functional connectivity, for now corr or cov
            supported
        :type fc_type: str
        :param time_as_row: whether shape of the array is time x ROI or
            transpossed
        :type time_as_row: bool
        :param atlas: brain parcellation atlas, if known
        :type atlas: str|tools.atlases.BaseAtlas|None
        """
        assert timeseries.ndim == 2, "Only 2D matrices supported"
        if fc_type == "corr":
            matrix = np.corrcoef(timeseries, rowvar=not time_as_row)
        elif fc_type == "cov":
            matrix = np.cov(timeseries, rowvar=not time_as_row)
        else:
            raise ValueError(f"Unknown FC type: {fc_type}")

        return cls(matrix, False, False, False, atlas=atlas)


class FiberLengthConnectivity(BaseConnectivity):
    """
    Matrix representing fiber lengths in segments.
    """

    connectivity_type = "fiber_length"
    label = "FibLen"

    def transform_to_delay_matrix(self, signal_velocity, segment_length=1.0):
        """
        Transform fiber length matrix into delay matrix. Delay matrix unit is
        ms.

        :param signal_velocity: signal velocity in m/s
        :type signal_velocity: float
        :param segment_length: length of a single segment in mm
        :type segment_length: float
        :return: initialized delay matrix with delays in ms
        :rtype: `DelayMatrix`
        """
        length_matrix = self.matrix * segment_length
        if signal_velocity > 0.0:
            matrix = length_matrix / signal_velocity
        elif signal_velocity == 0.0:
            matrix = np.zeros_like(self.matrix)
        else:
            raise ValueError("Cannot do negative signal velocity")
        return DelayMatrix(matrix, False, False, False, atlas=self.atlas)


class DelayMatrix(BaseConnectivity):
    """
    Matrix representing delay in ms.
    """

    connectivity_type = "delay"
    label = "Delay"

    @property
    def max_delay(self):
        """
        Return max delay.
        """
        return np.max(self.matrix)

    def unit_delay_diagonal(self, delay):
        """
        Insert given delay at the diagonal.

        :param delay: delay on diagonal, hence self-connection, in ms
        :type delay: float
        """
        np.fill_diagonal(self.matrix, delay)

    def transform_to_multiples_dt(self, dt, dt_in_seconds=True):
        """
        Transform delay matrix in ms to multiple of dts for simulation.

        :param dt: dt for the simulation, in ms or s
        :type dt: float
        :param dt_in_seconds: whether passed dt is in seconds
        :type dt_in_seconds: bool
        """
        if dt_in_seconds:
            dt *= 1000.0
        self.matrix = np.around(self.matrix / dt).astype(int)

import logging
import glob
import os

import numpy as np

import h5py
import scipy.io

from . import functions as func
from .collections import dotdict


class Dataset:
    """Dataset class."""

    def __init__(self, datasetName=None, normalizeCmats="max", fcd=False, subcortical=False):
        """
        Load the empirical data sets that are provided with `neurolib`.

        Right now, datasets work on a per-subject base. A dataset must be located
        in the `neurolib/data/datasets/` directory. Each subject's dataset
        must be in the `subjects` subdirectory of that folder. In each subject
        folder there is a directory called `functional` for time series data
        and `structural` the structural connectivity data.

        See `loadData.loadSubjectFiles()` for more details on which files are
        being loaded.

        The structural connectivity data (accessible using the attribute
        loadData.Cmat), can be normalized using the `normalizeCmats` flag.
        This defaults to "max" which normalizes the Cmat by its maxmimum.
        Other options are `waytotal` or `nvoxel`, which normalizes the
        Cmat by dividing every row of the matrix by the waytotal or
        nvoxel files that are provided in the datasets.

        Info: the waytotal.txt and the nvoxel.txt are files extracted from
        the tractography of DTI data using `probtrackX` from the `fsl` pipeline.

        Individual subject data is provided with the class attributes:
        self.BOLDs: BOLD timeseries of each individual
        self.FCs: Functional connectivity of BOLD timeseries

        Mean data is provided with the class attributes:
        self.Cmat: Structural connectivity matrix (for coupling strenghts between areas)
        self.Dmat: Fiber length matrix (for delays)
        self.BOLDs: BOLD timeseries of each area
        self.FCs: Functional connectiviy matrices of each BOLD timeseries

        :param datasetName: Name of the dataset to load
        :type datasetName: str
        :param normalizeCmats: Normalization method for the structural connectivity matrix. normalizationMethods = ["max", "waytotal", "nvoxel"]
        :type normalizeCmats: str
        :param fcd: Compute FCD matrices of BOLD data, defaults to False
        :type fcd: bool
        :param subcortical: Include subcortical areas from the atlas or not, defaults to False
        :type subcortical: bool

        """
        self.has_subjects = None
        if datasetName:
            self.loadDataset(datasetName, normalizeCmats=normalizeCmats, fcd=fcd, subcortical=subcortical)

    def loadDataset(self, datasetName, normalizeCmats="max", fcd=False, subcortical=False):
        """Load data into accessible class attributes.

        :param datasetName: Name of the dataset (must be in `datasets` directory)
        :type datasetName: str
        :param normalizeCmats: Normalization method for Cmats, defaults to "max"
        :type normalizeCmats: str, optional
        :raises NotImplementedError: If unknown normalization method is used
        """
        # the base directory of the dataset
        dsBaseDirectory = os.path.join(os.path.dirname(__file__), "..", "data", "datasets", datasetName)
        assert os.path.exists(dsBaseDirectory), f"Dataset {datasetName} not found in {dsBaseDirectory}."
        self.dsBaseDirectory = dsBaseDirectory
        self.data = dotdict({})

        # load all available subject data from disk to memory
        logging.info(f"Loading dataset {datasetName} from {self.dsBaseDirectory}.")
        self._loadSubjectFiles(self.dsBaseDirectory, subcortical=subcortical)
        assert len(self.data) > 0, "No data loaded."
        assert self.has_subjects

        self.Cmats = self._normalizeCmats(self.getDataPerSubject("cm"), method=normalizeCmats)
        self.Dmats = self.getDataPerSubject("len")

        # take the average of all
        self.Cmat = np.mean(self.Cmats, axis=0)

        self.Dmat = self.getDataPerSubject(
            "len",
            apply="all",
            apply_function=np.mean,
            apply_function_kwargs={"axis": 0},
        )
        self.BOLDs = self.getDataPerSubject("bold")
        self.FCs = self.getDataPerSubject("bold", apply_function=func.fc)

        if fcd:
            self.computeFCD()

        logging.info(f"Dataset {datasetName} loaded.")

    def computeFCD(self):
        logging.info("Computing FCD matrices ...")
        self.FCDs = self.getDataPerSubject("bold", apply_function=func.fcd, apply_function_kwargs={"stepsize": 10})

    def getDataPerSubject(
        self,
        name,
        apply="single",
        apply_function=None,
        apply_function_kwargs={},
        normalizeCmats="max",
    ):
        """Load data of a certain kind for all users of the current dataset

        :param name: Name of data type, i.e. "bold" or "cm"
        :type name: str
        :param apply: Apply function per subject ("single") or on all subjects ("all"), defaults to "single"
        :type apply: str, optional
        :param apply_function: Apply function on data, defaults to None
        :type apply_function: function, optional
        :param apply_function_kwargs: Keyword arguments of fuction, defaults to {}
        :type apply_function_kwargs: dict, optional
        :return: Subjectwise data, after function apply
        :rtype: list[np.ndarray]
        """
        values = []
        for subject, value in self.data["subjects"].items():
            assert name in value, f"Data type {name} not found in dataset of subject {subject}."
            val = value[name]
            if apply_function and apply == "single":
                val = apply_function(val, **apply_function_kwargs)
            values.append(val)

        if apply_function and apply == "all":
            values = apply_function(values, **apply_function_kwargs)
        return values

    def _normalizeCmats(self, Cmats, method="max", FSL_SAMPLES_PER_VOXEL=5000):
        # normalize per subject data
        normalizationMethods = [None, "max", "waytotal", "nvoxel"]
        if method not in normalizationMethods:
            raise NotImplementedError(
                f'"{method}" is not a known normalization method. Use one of these: {normalizationMethods}'
            )
        if method == "max":
            Cmats = [cm / np.max(cm) for cm in Cmats]
        elif method == "waytotal":
            self.waytotal = self.getDataPerSubject("waytotal")
            Cmats = [cm / wt for cm, wt in zip(Cmats, self.waytotal)]
        elif method == "nvoxel":
            self.nvoxel = self.getDataPerSubject("nvoxel")
            Cmats = [cm / (nv[:, 0] * FSL_SAMPLES_PER_VOXEL) for cm, nv in zip(Cmats, self.nvoxel)]
        return Cmats

    def _loadSubjectFiles(self, dsBaseDirectory, subcortical=False):
        """Dirty subject-wise file loader. Depends on the exact naming of all
        files as provided in the `neurolib/data/datasets` directory. Uses `glob.glob()`
        to find all files based on hardcoded file name matching.

        Can filter out subcortical regions from the AAL2 atlas.

        Info: Dirty implementation that assumes a lot of things about the dataset and filenames.

        :param dsBaseDirectory: Base directory of the dataset
        :type dsBaseDirectory: str
        :param subcortical: Filter subcortical regions from files defined by the AAL2 atlas, defaults to False
        :type subcortical: bool, optional
        """
        # check if there are subject files in the dataset
        if os.path.exists(os.path.join(dsBaseDirectory, "subjects")):
            self.has_subjects = True
            self.data["subjects"] = {}

            # data type paths, glob strings, dirty
            BOLD_paths_glob = os.path.join(dsBaseDirectory, "subjects", "*", "functional", "*rsfMRI*.mat")
            CM_paths_glob = os.path.join(dsBaseDirectory, "subjects", "*", "structural", "DTI_CM*.mat")
            LEN_paths_glob = os.path.join(dsBaseDirectory, "subjects", "*", "structural", "DTI_LEN*.mat")
            WAY_paths_glob = os.path.join(dsBaseDirectory, "subjects", "*", "structural", "waytotal*.txt")
            NVOXEL_paths_glob = os.path.join(dsBaseDirectory, "subjects", "*", "structural", "nvoxel*.txt")

            _ftypes = {
                "bold": BOLD_paths_glob,
                "cm": CM_paths_glob,
                "len": LEN_paths_glob,
                "waytotal": WAY_paths_glob,
                "nvoxel": NVOXEL_paths_glob,
            }

            for _name, _glob in _ftypes.items():
                fnames = glob.glob(_glob)
                # if there is none of this data type
                if len(fnames) == 0:
                    continue
                for f in fnames:
                    # dirty
                    subject = f.split(os.path.sep)[-3]
                    # create subject in dict if not present yet
                    if not subject in self.data["subjects"]:
                        self.data["subjects"][subject] = {}

                    # if the data for this type is not already loaded
                    if _name not in self.data["subjects"][subject]:
                        # bold, cm and len matrixes are provided as .mat files
                        if _name in ["bold", "cm", "len"]:
                            filter_subcotrical_axis = "both"
                            if _name == "bold":
                                key = "tc"
                                filter_subcotrical_axis = 0
                            elif _name == "cm":
                                key = "sc"
                            elif _name == "len":
                                key = "len"
                            # load the data
                            data = self.loadMatrix(f, key=key)
                            if not subcortical:
                                data = filterSubcortical(data, axis=filter_subcotrical_axis)
                            self.data["subjects"][subject][_name] = data
                        # waytotal and nvoxel files are .txt files
                        elif _name in ["waytotal", "nvoxel"]:
                            data = np.loadtxt(f)
                            if not subcortical:
                                data = filterSubcortical(data, axis=0)
                            self.data["subjects"][subject][_name] = data

    def loadMatrix(self, matFileName, key="", verbose=False):
        """Function to furiously load .mat files with scipy.io.loadmat.
        Info: More formats are supported but commented out in the code.

        :param matFileName: Filename of matrix to load
        :type matFileName: str
        :param key: .mat file key in which data is stored (example: "sc")
        :type key: str

        :return: Loaded matrix
        :rtype: numpy.ndarray
        """
        if verbose:
            print(f"Loading {matFileName}")
        matrix = scipy.io.loadmat(matFileName)
        if verbose:
            print("\tLoading using scipy.io.loadmat...")
            print(f"Keys: {list(matrix.keys())}")
        if key != "" and key in list(matrix.keys()):
            matrix = matrix[key]
            if verbose:
                print(f'\tLoaded key "{key}"')
        elif type(matrix) is dict:
            raise ValueError(f"Object is still a dict. Here are the keys: {matrix.keys()}")
        return matrix
        return 0


def filterSubcortical(a, axis="both"):
    """
    Filter out subcortical areas out of AAL2 atlas
    Indices from https://github.com/spunt/bspmview/blob/master/supportfiles/AAL2_README.txt
    Reminder: these are AAL indices, they start at 1!!!

    Hippocampus: 41 - 44
    Amygdala: 45-46
    Basal Ganglia: 75-80
    Thalamus: 81-82
    Cerebellum: 95-120

    :param a: Input (square) matrix with cortical and subcortical areas
    :type a: numpy.ndarray

    :return: Matrix without subcortical areas
    :rtype: numpy.ndarray
    """
    # with cerebellum indices
    # subcortical_index = np.array(list(range(40, 46)) + list(range(74, 82)) + list(range(94, 120)))
    # without cerebellum
    subcortical_index = np.array(list(range(40, 46)) + list(range(74, 82)))

    if axis == "both":
        a = np.delete(a, subcortical_index, axis=0)
        a = np.delete(a, subcortical_index, axis=1)
    else:
        a = np.delete(a, subcortical_index, axis=axis)
    return a

import glob
import os

import numpy as np

import h5py
import neurolib.utils.functions as func
import scipy.io


class Dataset:
    def __init__(self, datasetName=None):
        if datasetName:
            self.loadDataset(datasetName)

    def loadDataset(self, datasetName):
        """
        Load example empirical data sets that are provided with `neurolib`. 
        Datasets are loaded into class attributes: 
        self.Cmat: Structural connectivity matrix (for coupling strenghts between areas)
        self.Dmat: Fiber length matrix (for delays)
        self.BOLDs: BOLD timeseries of each area
        self.FCs: Functional connectiviy matrices of each BOLD timeseries

        :param datasetName: Name of the dataset to load
        :type datasetName: str
        """
        dsBaseDirectory = os.path.join(os.path.dirname(__file__), "..", "data", "datasets", datasetName)

        CmatFilename = os.path.join(dsBaseDirectory, "Cmat_avg.mat")
        self.Cmat = self.loadData(CmatFilename, key="sc", filter_subcortical=True)[0]  # structural connectivity matrix

        DmatFilename = os.path.join(dsBaseDirectory, "Dmat_avg.mat")
        self.Dmat = self.loadData(DmatFilename, key="len", filter_subcortical=True)[0]  # fiber length matrix

        BOLDFilenames = glob.glob(os.path.join(dsBaseDirectory, "BOLD/", "*_tc.mat"))  # BOLD timeseries

        self.BOLDs = self.loadData(BOLDFilenames, key="tc", filter_subcortical=True)
        self.FCs = self.loadData(BOLDFilenames, key="tc", filter_subcortical=True, apply_function=func.fc)
        # self.FCDs = self.loadData(BOLDFilenames, key="tc", filter_subcortical=True, apply_function=func.fcd, apply_function_kwargs={"stepsize": 10})

    def loadData(
        self,
        matrixFileNames,
        average=False,
        filter_subcortical=False,
        key="",
        apply_function=None,
        apply_function_kwargs={},
    ):
        """Loads matrices and applies operations on the matrices.
        
        :param matrixFileNames: List of filenames to load
        :type matrixFileNames: list[str]
        :param average: Take the average of all or not (consequently returns a list of a single matrix), defaults to False
        :type average: bool, optional
        :param filter_subcortical: Returns only cortical areas if set True, defaults to False
        :type filter_subcortical: bool, optional
        :param key: Key (string) in which data is stored in the .mat file, will be given to loadMatrix(), defaults to ""
        :type key: str, optional
        :param apply_function: Function to apply on loaded matrices, defaults to None
        :type apply_function: function, optional
        :param apply_function_kwargs: Keyword arguments for the applied function, defaults to {}
        :type apply_function_kwargs: dict, optional

        :return: List of matrices
        :rtype: list[np.ndarray]
        """
        # Handler if matrixFileNames is not a list but a str
        if isinstance(matrixFileNames, str):
            matrixFileNames = [matrixFileNames]

        matrices = []
        for matrixFileName in matrixFileNames:
            thisMat = self.loadMatrix(matrixFileName, key=key)
            if filter_subcortical:
                thisMat = filterSubcortical(thisMat)
            if apply_function:
                thisMat = apply_function(thisMat, **apply_function_kwargs)
            matrices.append(thisMat)

        if average:
            avgMatrix = np.zeros(matrices[0].shape)
            for cm in matrices:
                avgMatrix += cm
            avgMatrix /= len(matrices)
            return [avgMatrix]
        else:
            return matrices

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

        # bulletproof loading (old, maybe necessary)
        # if verbose:
        #     print("Loading {}".format(matFileName))
        # try:  # np.loadtxt
        #     if verbose:
        #         print("\tLoading using np.loadtxt...")
        #     matrix = np.loadtxt(matFileName)
        #     return matrix
        # except:
        #     pass
        # try:  # h5py.File
        #     matrix = h5py.File(matFileName, "r")
        #     if verbose:
        #         print("\tLoading using h5py.File...")
        #         print("Keys: {}".format(list(matrix.keys())))
        #     if key != "" and key in list(matrix.keys()):
        #         matrix = matrix[key].value
        #         if verbose:
        #             print('\tLoaded key "{}"'.format(key))
        #     elif type(matrix) is dict:
        #         raise ValueError("Object is still a dict. Here are the keys: {}".format(matrix.keys()))
        #     return matrix
        # except:  # scipy.io.loadmat
        #     matrix = scipy.io.loadmat(matFileName)
        #     if verbose:
        #         print("\tLoading using scipy.io.loadmat...")
        #         print("Keys: {}".format(list(matrix.keys())))
        #     if key != "" and key in list(matrix.keys()):
        #         matrix = matrix[key]
        #         if verbose:
        #             print('\tLoaded key "{}"'.format(key))
        #     elif type(matrix) is dict:
        #         raise ValueError("Object is still a dict. Here are the keys: {}".format(matrix.keys()))
        #     return matrix
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

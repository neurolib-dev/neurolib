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
        Load empirical data in aln2 parcellation
        Paramters:
            :param datasetName: Name of the dataset to load
        """
        dsBaseDirectory = os.path.join(
            os.path.dirname(__file__), "..", "data", "datasets", datasetName
        )

        CmatFilename = os.path.join(dsBaseDirectory, "Cmat_avg.mat")
        self.Cmat = self.loadData(
            CmatFilename, key="sc", filter_subcortical=True
        )  # structural connectivity matrix

        DmatFilename = os.path.join(dsBaseDirectory, "Dmat_avg.mat")
        self.Dmat = self.loadData(
            DmatFilename, key="len", filter_subcortical=True
        )  # fiber length matrix

        BOLDFilenames = glob.glob(
            os.path.join(dsBaseDirectory, "BOLD/", "*_tc.mat")
        )  # BOLD timeseries

        self.BOLDs = self.loadData(BOLDFilenames, key="tc", filter_subcortical=True)
        self.FCs = self.loadData(
            BOLDFilenames, key="tc", filter_subcortical=True, apply_function=func.fc
        )
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
        """
        Loads brain matrices provided filenames.

        Parameters:
        matrixFileNames (list): List of filenames to load
        average (bool): Take the average of all or not (consequently returns a single matric of a list of matrices)
        filter_subcortical (bool): Returns only cortical areas if set True
        key (str): Key (string) in which data is stored in the .mat file, will be given to loadMatrix()

        Returns:
        numpy.array: Single average matrix _or_ list of matrices
        """
        # print(matrixFileNames)
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

        if len(matrices) > 1:
            if average:
                avgMatrix = np.zeros(matrices[0].shape)
                for cm in matrices:
                    avgMatrix += cm
                avgMatrix /= len(matrices)
                return avgMatrix
            else:
                return matrices
        else:
            return matrices[0]

    def loadMatrix(self, matFileName, key="", verbose=False):
        """
        Function to load SC and FC .mat files of different formats.
        """
        if verbose:
            print("Loading {}".format(matFileName))
        try:  # FSL files:
            if verbose:
                print("\tLoading using np.loadtxt...")
            matrix = np.loadtxt(matFileName)
            return matrix
        except:
            pass
        try:  # LEAD DBS files:
            matrix = h5py.File(matFileName, "r")
            if verbose:
                print("\tLoading using h5py.File...")
                print("Keys: {}".format(list(matrix.keys())))
            if key != "" and key in list(matrix.keys()):
                matrix = matrix[key].value
                if verbose:
                    print('\tLoaded key "{}"'.format(key))
            elif type(matrix) is dict:
                raise ValueError(
                    "Object is still a dict. Here are the keys: {}".format(
                        matrix.keys()
                    )
                )
            return matrix
        except:  # Deco files
            matrix = scipy.io.loadmat(matFileName)
            if verbose:
                print("\tLoading using scipy.io.loadmat...")
                print("Keys: {}".format(list(matrix.keys())))
            if key != "" and key in list(matrix.keys()):
                matrix = matrix[key]
                if verbose:
                    print('\tLoaded key "{}"'.format(key))
            elif type(matrix) is dict:
                raise ValueError(
                    "Object is still a dict. Here are the keys: {}".format(
                        matrix.keys()
                    )
                )
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
    """

    # subcortical_index = np.array(list(range(40, 46)) + list(range(74, 82)) + list(range(94, 120)))
    subcortical_index = np.array(list(range(40, 46)) + list(range(74, 82)))

    if axis == "both":
        a = np.delete(a, subcortical_index, axis=0)
        a = np.delete(a, subcortical_index, axis=1)
    else:
        a = np.delete(a, subcortical_index, axis=axis)
    return a


# -----------------------------------------------
# Legacy (and duplicate) function definitions


def loadDataset(
    matrixFileNames,
    average=False,
    filter_subcortical=True,
    key="",
    apply_function=None,
    apply_function_kwargs={},
):
    """
    Loads brain matrices provided filenames.

    Parameters:
    matrixFileNames (list): List of filenames to load
    average (bool): Take the average of all or not (consequently returns a single matric of a list of matrices)
    filter_subcortical (bool): Returns only cortical areas if set True
    key (str): Key (string) in which data is stored in the .mat file, will be given to loadMatrix()

    Returns:
    numpy.array: Single average matrix _or_ list of matrices
    """
    # Handler if matrixFileNames is not a list but a str
    if isinstance(matrixFileNames, str):
        matrixFileNames = [matrixFileNames]

    matrices = []
    for matrixFileName in matrixFileNames:
        thisMat = loadMatrix(matrixFileName, key=key)
        if filter_subcortical:
            thisMat = filterSubcortical(thisMat)
        if apply_function:
            thisMat = apply_function(thisMat, **apply_function_kwargs)
        matrices.append(thisMat)

    if len(matrices) > 1:
        if average:
            avgMatrix = np.zeros(matrices[0].shape)
            for cm in matrices:
                avgMatrix += cm
            avgMatrix /= len(matrices)
            return avgMatrix
        else:
            return matrices
    else:
        return matrices[0]


# begin of function
def loadMatrix(matFileName, key="", verbose=False):
    """
    Function to load SC and FC .mat files of different formats.
    """
    if verbose:
        print("Loading {}".format(matFileName))
    try:  # FSL files:
        if verbose:
            print("\tLoading using np.loadtxt...")
        matrix = np.loadtxt(matFileName)
        return matrix
    except:
        pass
    try:  # LEAD DBS files:
        matrix = h5py.File(matFileName, "r")
        if verbose:
            print("\tLoading using h5py.File...")
            print("Keys: {}".format(list(matrix.keys())))
        if key != "" and key in list(matrix.keys()):
            matrix = matrix[key].value
            if verbose:
                print('\tLoaded key "{}"'.format(key))
        elif type(matrix) is dict:
            raise ValueError(
                "Object is still a dict. Here are the keys: {}".format(matrix.keys())
            )
        return matrix
    except:  # Deco files
        matrix = scipy.io.loadmat(matFileName)
        if verbose:
            print("\tLoading using scipy.io.loadmat...")
            print("Keys: {}".format(list(matrix.keys())))
        if key != "" and key in list(matrix.keys()):
            matrix = matrix[key]
            if verbose:
                print('\tLoaded key "{}"'.format(key))
        elif type(matrix) is dict:
            raise ValueError(
                "Object is still a dict. Here are the keys: {}".format(matrix.keys())
            )
        return matrix
    return 0

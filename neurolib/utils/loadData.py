import h5py
import numpy as np
import scipy.io

def loadDataset(matrixFileNames, average = False, filter_subcortical = True, key = '', \
                apply_function = None, apply_function_kwargs = {}):
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
def loadMatrix(matFileName, key='', verbose=False):
    """
    Function to load SC and FC .mat files of different formats.
    
    """
    if verbose:
        print("Loading {}".format(matFileName))
    try: # FSL files:
        if verbose:
            print("\tLoading using np.loadtxt...")
        matrix = np.loadtxt(matFileName)
        return matrix
    except:
        pass
    try: # LEAD DBS files:
        matrix = h5py.File(matFileName, 'r')
        if verbose:
            print("\tLoading using h5py.File...")
            print("Keys: {}".format(list(matrix.keys())))
        if key != '' and key in list(matrix.keys()):
            matrix = matrix[key].value
            if verbose:
                print("\tLoaded key \"{}\"".format(key))
        elif type(matrix) is dict:
            raise ValueError('Object is still a dict. Here are the keys: {}'.format(matrix.keys()) )
        return matrix
    except: # Deco files
        matrix = scipy.io.loadmat(matFileName)
        if verbose:
            print("\tLoading using scipy.io.loadmat...")
            print("Keys: {}".format(list(matrix.keys())))
        if key != '' and key in list(matrix.keys()):
            matrix = matrix[key]
            if verbose:
                print("\tLoaded key \"{}\"".format(key))
        elif type(matrix) is dict:
            raise ValueError('Object is still a dict. Here are the keys: {}'.format(matrix.keys()) )
        return matrix
    return 0


def filterSubcortical(a, axis = "both"):
    """
    Filter out subcortical areas out of aal2
    Hippocampus: 41 - 44
    Amygdala: 45-46
    Basal Ganglia: 75-80
    Thalamus: 81-82
    Cerebellum: 94-120
    """

    subcortical_index = np.array(list(range(41, 47)) + list(range(75, 83)) + list(range(94, 121)))

    if axis == "both":
        a = np.delete(a, subcortical_index, axis=0)
        a = np.delete(a, subcortical_index, axis=1)
    else:
        a = np.delete(a, subcortical_index, axis=axis)
    return a
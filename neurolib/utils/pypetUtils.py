import h5py
import pypet
import pathlib
import logging
import copy

from .collections import dotdict


def getTrajectorynamesInFile(filename):
    """
    Return a list of all pypet trajectory names in a a given hdf5 file.

    :param filename:  Name of the hdf file
    :type filename: str

    :return: List of strings containing the trajectory names
    :rtype: list[str]
    """
    assert pathlib.Path(filename).exists(), f"{filename} does not exist!"
    hdf = h5py.File(filename, "r")
    all_traj_names = list(hdf.keys())
    hdf.close()
    return all_traj_names


def loadPypetTrajectory(filename, trajectoryName):
    """Read HDF file with simulation results and return the chosen trajectory.

    :param filename: HDF file path
    :type filename: str

    :return: pypet trajectory
    """
    assert pathlib.Path(filename).exists(), f"{filename} does not exist!"
    logging.info(f"Loading results from {filename}")

    # if trajectoryName is not specified, load the most recent trajectory
    if trajectoryName == None:
        trajectoryName = getTrajectorynamesInFile(filename)[-1]
    logging.info(f"Analyzing trajectory {trajectoryName}")

    pypetTrajectory = pypet.Trajectory(trajectoryName, add_time=False)
    pypetTrajectory.f_load(trajectoryName, filename=filename, force=True)
    pypetTrajectory.v_auto_load = True
    return pypetTrajectory


def getRun(runId, pypetTrajectory, pypetShortNames=True):
    """Load the simulated data of a run and its parameters from a pypetTrajectory.

    :param runId: ID of the run
    :type runId: int
    :param pypetTrajectory: Pypet trajectory to get run from.
    :type pypetTrajectory: pypet.Trajectory
    :param pypetShortNames: Use pypet short names as keys for the results dictionary. Use if you are experiencing errors due to natural naming collisions.
    :type pypetShortNames: bool

    :return: Dictionary with simulated data and parameters of the run.
    :type return: dict
    """
    exploredParameters = pypetTrajectory.f_get_explored_parameters()
    niceParKeys = [p.split(".")[-1] for p in exploredParameters.keys()]

    pypetTrajectory.results[runId].f_load()
    result = pypetTrajectory.results[runId].f_to_dict(fast_access=True, short_names=pypetShortNames)
    pypetTrajectory.results[runId].f_remove()

    # convert to dotdict
    result = dotdict(result)

    # Postprocess result keys if pypet short names aren't used
    # Before: results.run_00000001.outputs.rates_inh
    # After: outputs.rates_inh
    if pypetShortNames == False:
        new_dict = {}
        for key, value in result.items():
            new_key = "".join(key.split(".", 2)[2:])
            new_dict[new_key] = result[key]
        result = copy.deepcopy(new_dict)

    # add parameters of this run
    result["params"] = {}

    for nicep, p in zip(niceParKeys, exploredParameters.keys()):
        result["params"][nicep] = exploredParameters[p].f_get_range()[runId]

    return result

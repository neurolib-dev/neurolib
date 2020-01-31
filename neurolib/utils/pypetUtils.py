import h5py
import pypet
import pathlib
import logging


def getTrajectorynamesInFile(filename):
    """
    Return a list of all pypet trajectories name saved in a a given hdf5 file.

    Parameter:
        :param filename:   Name of the hdf5 we want to explore

    Return:
        List of string containing the trajectory name
    """
    hdf = h5py.File(filename)
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

    trajLoaded = pypet.Trajectory(trajectoryName, add_time=False)
    trajLoaded.f_load(trajectoryName, filename=filename, force=True)
    trajLoaded.v_auto_load = True
    return trajLoaded

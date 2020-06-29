"""
Saving model output.
"""

import json

import pickle
from copy import deepcopy

import os
import numpy as np
import xarray as xr


def save_to_pickle(datafield, filename):
    """
    Save datafield to pickle file. Keep in mind that restoring a pickle
    requires that the internal structure of the types for the pickled data
    remain unchanged, o.e. not recommended for long-term storage.

    :param datafield: datafield or dataarray to save
    :type datafield: xr.Dataset|xr.DataArray
    :param filename: filename
    :type filename: str
    """
    assert isinstance(datafield, (xr.DataArray, xr.Dataset))
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    with open(filename, "wb") as handle:
        pickle.dump(datafield, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_to_netcdf(datafield, filename):
    """
    Save datafield to NetCDF. NetCDF cannot handle structured attributes,
    hence they are stripped and if there are some, they are saved as json
    with the same filename.

    :param datafield: datafield or dataarray to save
    :type datafield: xr.Dataset|xr.DataArray
    :param filename: filename
    :type filename: str
    """
    assert isinstance(datafield, (xr.DataArray, xr.Dataset))
    datafield = deepcopy(datafield)
    if not filename.endswith(".nc"):
        filename += ".nc"
    if datafield.attrs:
        attributes_copy = deepcopy(datafield.attrs)
        _save_attrs_json(attributes_copy, filename)
        datafield.attrs = {}
    datafield.to_netcdf(filename)


def _save_attrs_json(attrs, filename):
    """
    Save attributes to json.

    :param attrs: attributes to save
    :type attrs: dict
    :param filename: filename for the json file
    :type filename: str
    """

    def sanitise_attrs(attrs):
        sanitised = {}
        for k, v in attrs.items():
            if isinstance(v, list):
                sanitised[k] = [
                    sanitise_attrs(vv) if isinstance(vv, dict) else vv.tolist() if isinstance(vv, np.ndarray) else vv
                    for vv in v
                ]
            elif isinstance(v, dict):
                sanitised[k] = sanitise_attrs(v)
            elif isinstance(v, np.ndarray):
                sanitised[k] = v.tolist()
            else:
                sanitised[k] = v
        return sanitised

    filename = os.path.splitext(filename)[0] + ".json"
    with open(filename, "w") as handle:
        json.dump(sanitise_attrs(attrs), handle)

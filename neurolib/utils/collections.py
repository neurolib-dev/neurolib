"""
Collections of custom data structures and types.
"""
from dpath.util import search, delete
from collections import MutableMapping

DEFAULT_STAR_SEPARATOR = "."

FORWARD_REPLACE = {"*": "STAR"}
BACKWARD_REPLACE = {"STAR": "*"}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # now pickleable!!!
    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)


class star_dotdict(dotdict):
    """Support star notation in dotdict. Nested dicts are now treated as glob"""

    def __getitem__(self, attr):
        # if using star notation -> return dict of all keys that match
        if "*" in attr:
            return search(self, attr, separator=DEFAULT_STAR_SEPARATOR)
        # otherwise -> basic dict.get
        else:
            return dict.get(self, attr)

    def __setitem__(self, attr, val):
        # if using star notation -> search and set all keys matching
        if "*" in attr:
            for k, _ in search(self, attr, yielded=True, separator=DEFAULT_STAR_SEPARATOR):
                setattr(self, k, val)
        # otherwise -> just __setitem__
        else:
            dict.__setitem__(self, attr, val)

    def __delitem__(self, attr):
        # if using star notation -> use dpath's delete
        if "*" in attr:
            delete(self, attr, separator=DEFAULT_STAR_SEPARATOR)
        # otherwise -> just __delitem__
        else:
            dict.__delitem__(self, attr)


def _sanitize_keys(key, replace_dict):
    for k, v in replace_dict.items():
        key = key.replace(k, v)
    return key


def sanitize_dot_dict(dct, replace_dict=FORWARD_REPLACE):
    """
    Make all keys identifiers - replace "*" string for a word.
    """
    return {_sanitize_keys(k, replace_dict): v for k, v in dct.items()}


def unwrap_star_dotdict(dct, model, replaced_dict=False):
    """
    Unwrap star notation of parameters into full list of parameeters for a given
    model.
    E.g. params["*tau"] = 2.3 => params["mass1.tau"] = 2.3 and params["mass2.tau] = 2.3
    """
    # for each `k` that possibly contain stars get all key_u (unwrapped keys) from the star_dotdict
    if replaced_dict:
        return {
            key_u: v for k, v in dct.items() for key_u in list(model.params[_sanitize_keys(k, replaced_dict)].keys())
        }
    else:
        return {key_u: v for k, v in dct.items() for key_u in list(model.params[k].keys())}


def flatten_nested_dict(nested_dict, parent_key="", sep=DEFAULT_STAR_SEPARATOR):
    """
    Return flat dictionary from nested one using `sep` as separator between
    levels of keys.

    :param nested_dict: nested dictionary to flatten
    :type nested_dict: dict
    :param parent_key: parent key for the new key in flat dictionary
    :type parent_key: str
    :param sep: separator between levels of keys
    :type sep: str
    """
    items = []
    for k, v in nested_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flat_dict_to_nested(flat_dict, sep=DEFAULT_STAR_SEPARATOR):
    """
    Transform flat dictionary into nested one.

    :param flat_dict: flat dictionary with parameters, keys separated with `sep`
    :type flat_dict: dict
    :param sep: separator
    :type sep: str
    :return: nested dictionary
    :rtype: dict
    """

    def write_params_recurs(levels, out_dict, value_to_write):
        current_lookup = out_dict
        # all but last - that is actual key we want to write
        for level in levels[:-1]:
            # init empty dictionary if doesn't exists
            if level not in current_lookup:
                current_lookup[level] = {}
            # change lookup by level lower
            current_lookup = current_lookup[level]
        # write last level as value
        current_lookup[levels[-1]] = value_to_write

    nested_dict = {}
    for key, value in flat_dict.items():
        levels = key.split(sep)
        write_params_recurs(levels, nested_dict, value)

    return nested_dict

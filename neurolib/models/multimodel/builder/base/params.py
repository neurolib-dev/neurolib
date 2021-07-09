"""
Set of convenience functions for parameter handling in MultiModel.
"""

import numpy as np
import sympy as sp
from sympy.core import symbol


def count_float_params(param_dict):
    """
    Count number of float parameters in a dictionary, do not count noise
    parameters.
    """
    return len(
        {
            k: v
            for k, v in param_dict.items()
            if isinstance(v, (int, float))
            if not any(["input" in sp for sp in k.split(".")])
        }
    )


def float_params_to_vector_symbolic(param_dict):
    """
    Transforms float / array / int parameters to symbolic ones. Assumes flat
    dictionary with dot as a separator. Does not translate noise parameters.

    :param param_dict: dictionary with parameters and their values
    :type param_dict: dict
    :return: dictionary with parameters and symbols
    :rtype: dict
    """
    param_vec = sp.MatrixSymbol("param", n=count_float_params(param_dict), m=1)

    cnt = 0
    symbol_dict = {}
    for k, v in param_dict.items():
        splitted_key = k.split(".")
        if any(["input" in sp for sp in splitted_key]):
            continue
        if isinstance(v, (float, int)):
            symbol_dict[k] = param_vec[cnt, 0]
            cnt += 1
        elif isinstance(v, np.ndarray):
            symbol_dict[k] = sp.MatrixSymbol(k.replace(".", "DOT"), *v.shape)
        else:
            raise ValueError(f"Cannot handle {type(v)} type of {k}")
    return symbol_dict


def float_params_to_individual_symbolic(param_dict):
    """
    Transforms float / array / int parameters to symbolic ones. Assumes flat
    dictionary with dot as a separator. Does not translate noise parameters. All
    parameters are Symbols, compatible with symengine.
    """
    symbol_dict = {}
    for k, v in param_dict.items():
        splitted_key = k.split(".")
        if any(["input" in sp for sp in splitted_key]):
            continue
        if isinstance(v, (float, int)):
            symbol_dict[k] = sp.Symbol(k.replace(".", "DOT"))
        elif isinstance(v, np.ndarray):
            symbol_base_str = k.replace(".", "DOT")
            symbol_dict[k] = np.array(
                [sp.Symbol(f"{symbol_base_str}_{i}_{j}") for i in range(v.shape[0]) for j in range(v.shape[1])]
            ).reshape(v.shape)
        else:
            raise ValueError(f"Cannot handle {type(v)} type of {k}")
    return symbol_dict

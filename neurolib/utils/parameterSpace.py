"""
Class for representing a parameter space for exploration or optimization.
"""

from collections import namedtuple

import logging
import numpy as np
import pypet

from ..utils.collections import sanitize_dot_dict

SUPPORTED_KINDS = ["point", "bound", "grid", "sequence", "explicit", None]


class ParameterSpace:
    """
    Parameter space
    """

    def __init__(self, parameters, parameterValues=None, kind=None, allow_star_notation=False):
        """
        Initialize parameter space. Parameter space can be initialized in two ways:
        Either a `parameters` is a dictionary of the form `{"parName1" : [0, 1, 2], "parName2" : [3, 4]}`,
        or `parameters` is a list of names and `parameterValues` are values of each parameter.

        :param parameters: parameter dictionary or list of names of parameters e.g. `['x', 'y']`
        :type parameters: `dict, list[str, str]`
        :param parameterValues: list of parameter values (must be floats) e.g. `[[x_min, x_max], [y_min, y_max], ...]`
        :type parameterValues: `list[list[float, float]]`
        :param kind: string describing the kind of parameter space:
            - `point`: a single point in parameter space
            - `bound`: a bound in parameter space, i.e. two values per parameter
            - `grid`: a cartesian product over parameters
            - `sequence`: a sequence of univariate parameter changes - only one will change at the time, other
                parameters will stay as default
            - `explicit`: explicitely define a parameter space, i.e lists of all parameters have to have the same length
            - None: parameterSpace tries to auto-detect the correct kind
        :type kind: str
        :param allow_star_notation: whether to allow star notation in parameter names - MultiModel
        :type allow_star_notation: bool
        """
        assert kind in SUPPORTED_KINDS
        self.kind = kind
        self.parameters = parameters
        self.star = allow_star_notation
        # in case a parameter dictionary was given
        if parameterValues is None:
            assert isinstance(
                parameters, dict
            ), "Parameters must be a dict, if no values are given in `parameterValues`"
        else:
            # check if all names are strings
            assert np.all([isinstance(pn, str) for pn in parameters]), "Parameter names must all be strings."
            # check if all parameter values are lists
            assert np.all([isinstance(pv, (list, tuple)) for pv in parameterValues]), "Parameter values must be a list."
            parameters = self._parameterListsToDict(parameters, parameterValues)

        self.parameters = self._processParameterDict(parameters)
        self.parameterNames = list(self.parameters.keys())
        self.parameterValues = list(self.parameters.values())

        # let's create a named tuple of the parameters
        # Note: evolution.py implementation relies on named tuples
        self.named_tuple_constructor = namedtuple("ParameterSpace", sanitize_dot_dict(parameters))
        self.named_tuple = self.named_tuple_constructor(*self.parameterValues)

        # set attributes of this class to make it accessible
        for i, p in enumerate(self.parameters):
            setattr(self, p, self.parameterValues[i])

    def __str__(self):
        """Print the named_tuple object"""
        return str(self.parameters)

    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value
        self._processParameterDict(self.parameters)

    def dict(self):
        """Returns the parameter space as a dicitonary of lists.
        :rtype: dict
        """
        return self.parameters

    def get_parametrization(self):
        assert self.kind is not None
        if self.kind in ["point", "bound", "explicit"]:
            # check same length
            it = iter(self.parameters.values())
            length = len(next(it))
            assert all(len(l) == length for l in it)
            # just return as dict
            return self.parameters
        elif self.kind == "grid":
            # cartesian product
            return pypet.cartesian_product(self.parameters)
        elif self.kind == "sequence":
            # return as sequence
            return self._inflate_to_sequence(self.parameters)

    @staticmethod
    def _inflate_to_sequence(param_dict):
        """
        Inflate dict of parameters to a sequence of same length, using None as
        placeholder when a particular parameter should not change.
        {"a": [1, 2], "b": [3, 4, 5]} ->
            {"a": [1, 2, None, None, None], "b": [None, None, 3, 4, 5]}
        """
        return {
            k: [None] * sum([len(tmp) for tmp in list(param_dict.values())[:i]])
            + v
            + [None] * sum([len(tmp) for tmp in list(param_dict.values())[i + 1 :]])
            for i, (k, v) in enumerate(param_dict.items())
        }

    def getRandom(self, safe=False):
        """This function returns a random single parameter from the whole space
        in the form of { "par1" : 1, "par2" : 2}.

        This function is used by neurolib/optimize/exploarion.py
        to add parameters of the space to pypet (for initialization)

        :param safe: Return a "safe" parameter or the original. Safe refers to
        returning python floats, not, for example numpy.float64 (necessary for pypet).
        ;type safe: bool
        """
        randomPar = {}
        if safe:
            for key, value in self.parameters.items():
                random_value = np.random.choice(value)
                if isinstance(random_value, np.float64):
                    random_value = float(random_value)
                elif isinstance(random_value, np.int64):
                    random_value = int(random_value)
                randomPar[key] = random_value
        else:
            for key, value in self.parameters.items():
                randomPar[key] = np.random.choice(value)
        return randomPar

    @property
    def lowerBound(self):
        """Returns lower bound of all parameters as a list"""
        return [np.min(p) for p in self.parameterValues]

    @property
    def upperBound(self):
        """Returns upper bound of all parameters as a list"""
        return [np.max(p) for p in self.parameterValues]

    @property
    def ndims(self):
        """Number of dimensions (parameters)"""
        return len(self.parameters)

    @staticmethod
    def _validate_single_bound(single_bound):
        """
        Validate single bound.
        :param single_bound: single coordinate bound to validate
        :type single_bound: list|tuple
        """
        assert isinstance(
            single_bound, (list, tuple)
        ), "An error occured while validating the ParameterSpace of kind 'bound': Pass parameter bounds as a list or tuple!"
        assert (
            len(single_bound) == 2
        ), "An error occured while validating the ParameterSpace of kind 'bound': Only two bounds (min and max) are allowed"
        assert (
            single_bound[1] > single_bound[0]
        ), "An error occured while validating the ParameterSpace of kind 'bound': Minimum parameter value can't be larger than the maximum!"

    def _validate_param_bounds(self, param_bounds):
        """
        Validate param bounds.
        :param param_bounds: parameter bounds to validate
        :type param_bounds: list|None
        """
        assert param_bounds is not None
        assert isinstance(param_bounds, (list, tuple))
        # check every single parameter bound
        for single_bound in param_bounds:
            self._validate_single_bound(single_bound)

    def _processParameterDict(self, parameters):
        """Processes all parameters and do checks. Determine the kind of the parameter space.
        :param parameters: parameter dictionary
        :type param: dict

        :retun: processed parameter dictionary
        :rtype: dict
        """

        # convert all parameter arrays into lists
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                assert len(value.shape) == 1, f"Parameter {key} is not one-dimensional."
                value = value.tolist()
                parameters[key] = value

        # auto detect the parameter kind
        if self.kind is None:
            for key, value in parameters.items():
                # auto detect what kind of space we have
                # kind = "point" is a single point in parameter space, one value only
                # kind = "bound" is a bounded parameter space with 2 values: min and max
                # kind = "grid" is a grid space with as many values on each axis as wished

                # first, we assume grid
                self.kind = "grid"
                parameterLengths = [len(value) for key, value in parameters.items()]
                # if all parameters have the same length
                if parameterLengths.count(parameterLengths[0]) == len(parameterLengths):
                    if parameterLengths[0] == 1:
                        self.kind = "point"
                    elif parameterLengths[0] == 2:
                        self.kind = "bound"
            logging.info(f'Assuming parameter kind "{self.kind}"')

        # do some kind-specific tests
        if self.kind == "bound":
            # check the boundaries
            self._validate_param_bounds(list(parameters.values()))

        # set all parameters as attributes for easy access
        for key, value in parameters.items():
            setattr(self, key, value)

        return parameters

    def _parameterListsToDict(self, keys, values):
        parameters = {}
        assert len(keys) == len(values), "Names and values of parameters are not same length."
        for key, value in zip(keys, values):
            parameters[key] = value
        return parameters

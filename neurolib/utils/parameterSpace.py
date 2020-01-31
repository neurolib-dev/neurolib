import collections


class ParameterSpace:
    """Paremter space
    """

    def __init__(self, parameter_names, parameter_bounds):
        """Initialize parameter space
        
        :param parameter_names: list of names of parameters e.g. ['x', 'y']
        :type parameter_names: list[str, str]
        
        :param parameter_bounds: list of parameter bounds e.g. [[x_min, x_max], [y_min, y_max], ...]
        :type parameter_bounds: list[list[float, float]]
        """
        # check parameters
        self._validate_param_bounds(parameter_bounds)
        # check parameter names
        for s in parameter_names:
            assert isinstance(s, str), f"Parameter name {s} is not a string!"

        self.parameter_names = parameter_names
        self.parameter_bounds = parameter_bounds

        # let's create a named tuple of the parameters
        # Note: evolution.py implementation relies on named tuples
        self.named_tuple_constructor = collections.namedtuple(
            "ParameterSpace", parameter_names
        )
        self.named_tuple = self.named_tuple_constructor(*parameter_bounds)

        # set attributes of this class to make it accessible
        for i, p in enumerate(self.parameter_names):
            setattr(self, p, self.parameter_bounds[i])

    def __str__(self):
        """Print the named_tuple object
        """
        return str(self.named_tuple)

    @property
    def ndims(self):
        """Number of dimensions (parameters)
        """
        return len(self.parameter_names)

    @staticmethod
    def _validate_single_bound(single_bound):
        """
        Validate single bound.
        :param single_bound: single coordinate bound to validate
        :type single_bound: list|tuple
        """
        assert isinstance(
            single_bound, (list, tuple)
        ), "Pass parameter bounds as a list or tuple!"
        assert len(single_bound) == 2, "Only two bounds (min and max) are allowed"
        assert (
            single_bound[1] > single_bound[0]
        ), "Minimum parameter value can't be larger than the maximum!"

    def _validate_param_bounds(self, param_bounds):
        """
        Validate param bounds.
        :param param_bounds: parameter bounds to validate
        :type param_bounds: list|None
        """
        assert param_bounds is not None
        assert isinstance(param_bounds, (list, tuple))
        [self._validate_single_bound(single_bound) for single_bound in param_bounds]


"""
Collections of custom data structures and types.
"""


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

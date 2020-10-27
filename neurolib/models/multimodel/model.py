from ..model import Model


class MultiModel(Model):
    """
    Base for all MultiModel's i.e. heterogeneous networks or network nodes built
    using model builder.
    """

    @classmethod
    def init_node(cls):
        pass

    @classmethod
    def init_network(cls):
        pass

    def __init__(self):
        pass

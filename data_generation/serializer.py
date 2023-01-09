import abc
import json

class ArgSerializable(abc.ABC):

    def __init__(self, args: dict):
        """

        :param args: All keyword arguments the constructor of the subclass took. This is used for serialization
        """
        self.args = args

    def __dict__(self):
        # Ugly quick fix
        kwargs = json.loads(json.dumps(self.args, default=lambda o: o.__dict__()))
        return dict(_type=self.__class__.__name__, args=kwargs)

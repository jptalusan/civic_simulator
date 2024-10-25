from abc import ABC, abstractmethod

class BaseRouter(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_route(self, o, d, weight):
        pass

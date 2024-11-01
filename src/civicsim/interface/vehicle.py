from abc import ABC, abstractmethod


class Vehicle(ABC):
    def __init__(self, capacity=None, current_stop=None) -> None:
        super().__init__()
        self.id = None
        self.capacity = capacity
        self.current_stop = current_stop
        self.next_stop = None
        self.time = None
        self.current_point = None
        self.current_route_linestring = None
        self.load = 0

    @abstractmethod
    def pickup(self):
        pass


class ExpressBus(Vehicle):
    def __init__(self, capacity, current_stop) -> None:
        super().__init__(capacity, current_stop)

    def pickup(self):
        return super().pickup()

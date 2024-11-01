import src.civicsim.common.constants as constants


class Event:
    def __init__(self, event_type, time, type_specific_information=None):
        """
        time: minutes
        """
        self.event_type = event_type
        self.time = time
        self.type_specific_information = type_specific_information

    def __str__(self):
        if self.event_type == constants.EVENT_PICKUP:
            event_type = "EVENT_PICKUP"
        elif self.event_type == constants.EVENT_DROPOFF:
            event_type = "EVENT_DROPOFF"
        elif self.event_type == constants.EVENT_TIMER:
            event_type = "EVENT_TIMER"
        if self.type_specific_information:
            return f"{self.time},{event_type},{self.type_specific_information}"
        return f"{self.time},{self.event_type}"

    def __repr__(self):
        if self.event_type == constants.EVENT_PICKUP:
            event_type = "EVENT_PICKUP"
        elif self.event_type == constants.EVENT_DROPOFF:
            event_type = "EVENT_DROPOFF"
        elif self.event_type == constants.EVENT_TIMER:
            event_type = "EVENT_TIMER"
        if self.type_specific_information:
            return f"{self.time},{event_type},{self.type_specific_information}"
        return f"{self.time},{self.event_type}"

# Easier to serialize global constants as fake enums
TYPE_OBSTACLE = 1
TYPE_CAR = 2
TYPE_SEMI = 3

STATUS_GO = 4
STATUS_STOP = 5
STATUS_BREAK = 6

class ObjectInformation:
    def __init__(self, object_type, status, x, y, heading):
        self.object_type = object_type
        self.status = status
        self.x = x
        self.y = y
        self.heading = heading

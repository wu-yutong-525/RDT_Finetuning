from enum import Enum

class Command(Enum):
    STOP = 0
    EMERGENCY_STOP = 1
    MOVEJ = 2
    SERVOJ = 3
    SERVOL = 4
    GRASP = 5
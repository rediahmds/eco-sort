from enum import Enum


class BlynkPins(Enum):
    """A list of constansts that represents virtual pins
    used in this project specifically.

    ### Available pins and functionality:

    `V0`: prediction result

    `V1`: servo controller
    """

    V0 = "v0"
    V1 = "v1"

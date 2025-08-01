from enum import Enum


class BlynkPins(Enum):
    """A list of constansts that represents virtual pins
    used in this project specifically.

    ### Available pins and functionality:

    `V0`: prediction result

    `V1`: servo controller

    `V2`: organic bin level indicator

    `V3`: recycle/Non-organic bin level indicator

    `V4`: methane sensor value
    """

    V0 = "v0"
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    V4 = "v4"
    V7 = "v7"

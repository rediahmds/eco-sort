from enum import Enum


class BlynkServer(Enum):
    """A list of constants that represents Blynk's server address."""

    SGP = "sgp1.blynk.cloud"
    NY = "ny3.blynk.cloud"
    DEFAULT = "blnk.cloud"

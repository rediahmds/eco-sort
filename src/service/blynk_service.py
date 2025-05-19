import requests
from static.blynk.blynk_server import BlynkServer
from static.blynk.blynk_pins import BlynkPins
import traceback

# Token can be used in both device, dont need to be unique for each


class BlynkService:
    def __init__(self, token: str, server: str | BlynkServer = BlynkServer.SGP):
        self.token = token

        if isinstance(server, BlynkServer):
            self.base_url = f"https://{server.value}/external/api"
        else:
            self.base_url = f"https://{server}/external/api"

    def getDatastreamValue(self, virtual_pin: BlynkPins):
        pin = virtual_pin.value
        try:
            req = requests.get(f"{self.base_url}/get?token={self.token}&{pin}")

        except Exception as e:
            print(
                f"Unknown error while getting data stream value for pin: {virtual_pin}"
            )

    def updateDatastreamValue(self, virtual_pin: BlynkPins, value: str):
        pin = virtual_pin.value
        params = {"token": self.token, pin: value}
        try:
            requests.get(f"{self.base_url}/update", params=params)

        except Exception as e:
            print(f"Failed to write data to pin: {virtual_pin}")
            traceback.print_exc()

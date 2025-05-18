import requests
from static.blynk.blynk_server import BlynkServer
from static.blynk.blynk_pins import BlynkPins

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

    def postDatastreamValue(self, virtual_pin: BlynkPins):
        pin = virtual_pin.value
        try:
            res = requests.post()
        except Exception as e:
            print(f"Failed to write data to pin: {virtual_pin}")

import requests
from static.blynk.blynk_server import BlynkServer
from static.blynk.blynk_pins import BlynkPins
import traceback


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
            response = requests.get(f"{self.base_url}/get?token={self.token}&{pin}")
            print(f"Response data: \n{response.text}")
        except Exception as e:
            print(
                f"Unknown error while getting data stream value for pin: {virtual_pin}"
            )
            traceback.print_exc()

    def updateDatastreamValue(self, virtual_pin: BlynkPins, value: str) -> bool:
        pin = virtual_pin.value
        params = {"token": self.token, pin: value}
        try:
            response = requests.get(f"{self.base_url}/update", params=params)
            status_code = response.status_code

            return True if status_code >= 200 or status_code < 300 else False
        except Exception as e:
            print(f"Failed to write data to pin: {virtual_pin}")
            print(f"Reason: {e}")
            traceback.print_exc()

            return False

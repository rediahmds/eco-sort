import requests
from src.static.blynk_server import BlynkServer


class BlynkService:
    def __init__(self, token: str, server: str | BlynkServer = BlynkServer.SGP):
        self.token = token

        if isinstance(server, BlynkServer):
            self.base_url = f"https://{server.value}/external/api"
        else:
            self.base_url = f"https://{server}/external/api"

    def getDatastreamValue(self, virtual_pin: str):
        try:
            req = requests.get(f"{self.base_url}/get?token={self.token}&{virtual_pin}")
        
        except Exception as e:
            print(f"Unknown error while getting data stream value for pin: {virtual_pin}")

import json
import requests

class TFServingConector():

    def __init__(self, host='localhost', port='8501'):
        self.host=host
        self.port=port
        self.headers={'Content-Type': "application/json", 'cache-control': "no-cache"}
        self.url="http://{}:{}/v1/models/trex:predict".format(host, port)

    def post(self, instances):
        load = {"instances": instances}
        payload = json.dumps(load)
        response = requests.request("POST",
                                    self.url,
                                    data=payload,
                                    headers=self.headers)
        return response.text
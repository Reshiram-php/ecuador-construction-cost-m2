import sys

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/predict"

building = {
    "codreg": "3", "codprovf": "24", "codcantf": "2101", "careaur": "2",
    "cimi": "120", "piso": "152", "estru": "125", "entrepisos": "5", "pared": "126", "cubi": "135",
    "ctipobr": "1", "cores": "160", "coams": "0", "cdispuso": "1", "cdisprfi": "12",
    "espverd": "2", "asagua": "1", "asluz": "1", "asalcant": "1", "cintern": "1", "astelf": "1", "aedif": "5005",
    "tocup": "5011", "fideic": "2", "casaedif": "1", "tipfinan": "6002", "valterr": "1",
    "cares": 71, "carnres": 0, "csute": 360, "caesv": 289, "carpat": 0,
    "nupical": 1, "cperm": 1, "cnuvical": 1, "nucucal": 7, "nudocal": 2,
}

response = requests.post(url, json=building, timeout=10)
print(response.status_code, response.json())


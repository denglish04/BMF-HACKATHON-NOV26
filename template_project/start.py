import requests

BASE = "https://api.ejanalysis.com"

resp = requests.post(
    f"{BASE}/data",
    json={
        "lat": 34.0522,
        "lon": -118.2437,
        "buffer": 3,  # miles
    },
    timeout=120,
)
print(resp.status_code)
print(resp.text[:500])   # look at the raw response before parsing
resp.raise_for_status()
data = resp.json()
print(data)

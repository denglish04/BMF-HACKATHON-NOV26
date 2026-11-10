import requests
import pandas as pd

BASE = "https://api.ejanalysis.com"

resp = requests.post(
    f"{BASE}/data",
    params={"fips": "06037", "scale": "county"},
    timeout=120,
)
resp.raise_for_status()

df = pd.DataFrame(resp.json())

# Drop the HTML link columns
df = df.drop(columns=["EJAM Report", "EJSCREEN Map"], errors="ignore")

print(df.shape)
print(df.columns.tolist())      # see everything the API returns
print(df.T.head(30))            # transposed: easier to read with one row

df.to_csv("la_county.csv", index=False)

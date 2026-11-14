import requests, pathlib, pandas as pd
from tqdm import tqdm

BASE = "https://api.ejanalysis.com"
cache = pathlib.Path("cache"); cache.mkdir(exist_ok=True)

def fetch_bg(fips):
    out = cache / f"{fips}_bg.json"
    if out.exists():                      # cached: no API call
        return pd.read_json(out, dtype={"ejam_uniq_id": str})

    tmp = out.with_suffix(".part")
    with requests.post(f"{BASE}/data",
                       params={"fips": fips, "scale": "blockgroup"},
                       stream=True, timeout=(10, 300)) as r:
        print(fips, r.status_code)
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=fips) as bar:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(out)                       # only counts as cached if the download finished
    return pd.read_json(out, dtype={"ejam_uniq_id": str})

if __name__ == "__main__":
    df = fetch_bg("17097")                # Lake County, IL
    print(df.shape)                       # expect (425, 679)
    print(df["ejam_uniq_id"].head())

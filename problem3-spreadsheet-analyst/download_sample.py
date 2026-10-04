from __future__ import annotations

import argparse
from pathlib import Path
import requests

URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=50000)
    p.add_argument("--out", type=Path, default=Path("data/nyc311_sample.csv"))
    a = p.parse_args()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    params = {"$limit": min(max(a.rows, 100), 100000), "$order": "created_date DESC"}
    r = requests.get(URL, params=params, timeout=120)
    r.raise_for_status()
    a.out.write_bytes(r.content)
    print(f"Saved {a.out}")

if __name__ == "__main__":
    main()

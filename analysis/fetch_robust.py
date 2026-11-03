"""
Robust IMD downloader: POST per year with timeouts and retries, limited
concurrency, then clip to Kerala with imdlib's own parser.

Usage (from the data dir): python fetch_robust.py [--workers 3] [--end 2025]
"""
import argparse
import calendar
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imdlib
import requests

URLS = {
    "rain": ("https://imdpune.gov.in/cmpg/Griddata/rainfall.php", "rain", ".grd", (129, 135)),
    "tmax": ("https://imdpune.gov.in/cmpg/Griddata/maxtemp.php", "maxtemp", ".GRD", (31, 31)),
    "tmin": ("https://imdpune.gov.in/cmpg/Griddata/mintemp.php", "mintemp", ".GRD", (31, 31)),
}
START = {"rain": 1901, "tmax": 1951, "tmin": 1951}
SENTINEL = {"rain": -999.0, "tmax": 99.9, "tmin": 99.9}
LAT, LON = slice(7.75, 13.25), slice(74.5, 77.75)
RAW, CLIP = Path("_raw2"), Path("_clipped")


def download(var, year, tries=5):
    url, key, ext, (nlat, nlon) = URLS[var]
    days = 366 if calendar.isleap(year) else 365
    expected = days * nlat * nlon * 4
    dest = RAW / var / f"{year}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == expected:
        return dest
    last = None
    for attempt in range(tries):
        try:
            r = requests.post(url, data={key: year}, timeout=(20, 240))
            r.raise_for_status()
            if len(r.content) == 0:
                return None  # not published
            if len(r.content) != expected:
                raise ValueError(f"size {len(r.content)} != {expected}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{var} {year}: {last}")


def clip(var, year):
    out = CLIP / f"{var}_{year}.nc"
    if out.exists():
        return "cached"
    p = download(var, year)
    if p is None:
        return "unpublished"
    ds = imdlib.open_data(var, year, year, "yearwise", str(RAW)).get_xarray()
    da = ds[var].sel(lat=LAT, lon=LON)
    da = da.where(da != SENTINEL[var]).where(da > -900)
    tmp = CLIP / f".{var}_{year}.nc.part"
    da.to_dataset(name=var).to_netcdf(tmp)
    tmp.replace(out)
    p.unlink()
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--end", type=int, default=2025)
    a = ap.parse_args()
    CLIP.mkdir(exist_ok=True)
    jobs = [(v, y) for v in ("rain", "tmax", "tmin") for y in range(START[v], a.end + 1)]
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(clip, v, y): (v, y) for v, y in jobs}
        done = 0
        for f in as_completed(futs):
            v, y = futs[f]
            done += 1
            try:
                status = f.result()
            except Exception as exc:  # noqa: BLE001
                status = f"FAIL {exc}"
            print(f"[{done}/{len(jobs)} {time.time()-t0:.0f}s] {v} {y} {status}", flush=True)
    shutil.rmtree(RAW, ignore_errors=True)


if __name__ == "__main__":
    main()

"""Merge per-year clipped files into one NetCDF per variable. Run from data dir."""
import sys
from pathlib import Path

import xarray as xr

STEM = {"rain": "kerala_rain_0p25", "tmax": "kerala_tmax_1p0", "tmin": "kerala_tmin_1p0"}
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
for var, stem in STEM.items():
    files = sorted(Path("_clipped").glob(f"{var}_*.nc"))
    if not files:
        continue
    for old in out_dir.glob(f"{stem}_*.nc"):
        old.unlink()
    ds = xr.concat([xr.open_dataset(f).load() for f in files], dim="time").sortby("time")
    y0, y1 = int(ds.time.dt.year.min()), int(ds.time.dt.year.max())
    out = out_dir / f"{stem}_{y0}_{y1}.nc"
    ds.to_netcdf(out, encoding={var: {"zlib": True, "complevel": 4}})
    print(var, len(files), "files ->", out.name, f"{out.stat().st_size/1e6:.1f} MB")

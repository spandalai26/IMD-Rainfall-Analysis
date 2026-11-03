"""
Seasonal ETCCDI-style rainfall and temperature indices for Kerala from IMD
gridded data, computed two ways:

  cell  - index computed in every grid cell, then area-weighted mean over Kerala
  state - daily values averaged over Kerala first, index computed on that series

Seasons (IMD): winter JF, pre-monsoon MAM, monsoon JJAS, post-monsoon OND.
Trend: Sen's slope + Hamed-Rao modified Mann-Kendall (handles autocorrelation).

Usage: python analyze_kerala.py <data_dir> <kerala.geojson> <out.json>
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pymannkendall as mk
import xarray as xr
from shapely.geometry import box

SEASONS = {
    "winter": [1, 2],
    "premonsoon": [3, 4, 5],
    "monsoon": [6, 7, 8, 9],
    "postmonsoon": [10, 11, 12],
}
WET = 1.0  # mm, ETCCDI wet-day threshold
EARLY, RECENT = (1901, 1950), (1991, 2025)
T_EARLY, T_RECENT = (1951, 1980), (1996, 2025)
T_BASE = (1961, 1990)  # baseline for percentile-based hot-day index


# ---------------------------------------------------------------- geometry
def overlap_fraction(lat, lon, poly, res):
    """Fraction of each grid cell's area inside the polygon."""
    frac = np.zeros((lat.size, lon.size))
    h = res / 2
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            cell = box(lo - h, la - h, lo + h, la + h)
            if cell.intersects(poly):
                frac[i, j] = cell.intersection(poly).area / cell.area
    w = frac * np.cos(np.deg2rad(lat))[:, None]
    return xr.DataArray(w, coords={"lat": lat, "lon": lon}, dims=("lat", "lon")), \
        xr.DataArray(frac, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def state_weights(da, poly, res, min_frac):
    w, frac = overlap_fraction(da.lat.values, da.lon.values, poly, res)
    has_data = da.notnull().mean("time") > 0.9
    w = w.where((frac >= min_frac) & has_data, 0.0)
    return w


def wmean(da, w):
    """Weighted mean over lat/lon, skipping NaN."""
    valid = da.notnull()
    num = (da.fillna(0) * w).sum(("lat", "lon"))
    den = (valid * w).sum(("lat", "lon"))
    return num / den


# ---------------------------------------------------------------- indices
def max_run(mask: np.ndarray) -> np.ndarray:
    """Longest run of True along axis 0. mask shape (t, ...)."""
    run = np.zeros(mask.shape[1:], dtype=np.int32)
    best = np.zeros_like(run)
    for t in range(mask.shape[0]):
        run = np.where(mask[t], run + 1, 0)
        best = np.maximum(best, run)
    return best


def rain_indices(pr: xr.DataArray) -> dict:
    """pr: (time, ...) for one season of one year. Returns dict of arrays."""
    a = pr.values
    valid = ~np.isnan(a)
    wet = (a >= WET) & valid
    dry = (a < WET) & valid
    a0 = np.where(valid, a, 0.0)
    nwet = wet.sum(0)
    prcptot = np.where(wet, a0, 0).sum(0)
    rx5 = np.full(a.shape[1:], np.nan)
    if a.shape[0] >= 5:
        c = np.cumsum(np.vstack([np.zeros((1,) + a.shape[1:]), a0]), axis=0)
        rx5 = (c[5:] - c[:-5]).max(0)
    out = {
        "wet_days": nwet.astype(float),
        "cwd": max_run(wet).astype(float),
        "cdd": max_run(dry).astype(float),
        "prcptot": prcptot,
        "rx1day": a0.max(0),
        "rx5day": rx5,
        "r20mm": ((a0 >= 20) & valid).sum(0).astype(float),
        "sdii": np.where(nwet > 0, prcptot / np.maximum(nwet, 1), np.nan),
    }
    allnan = ~valid.any(0)
    return {k: np.where(allnan, np.nan, v) for k, v in out.items()}


def temp_indices(tx, tn, tx90) -> dict:
    """tx, tn: (time, ...) one season one year; tx90 threshold (...)."""
    x, n = tx.values, tn.values
    with np.errstate(invalid="ignore"):
        return {
            "tmax_mean": np.nanmean(x, 0),
            "tmin_mean": np.nanmean(n, 0),
            "dtr": np.nanmean(x - n, 0),
            "txx": np.nanmax(x, 0),
            "tnx": np.nanmax(n, 0),
            "tx90p": 100 * np.nanmean(x > tx90, 0),
        }


def season_slices(time: pd.DatetimeIndex, months):
    """Yield (year, boolean index) for each year's season."""
    years = sorted(set(time.year))
    for y in years:
        sel = (time.year == y) & np.isin(time.month, months)
        if sel.sum() > 0:
            yield y, sel


def compute_rain(pr, w):
    time = pd.DatetimeIndex(pr.time.values)
    state_series = wmean(pr, w)  # (time,)
    rows = []
    cell_maps = {}  # season -> index -> (years, lat, lon)
    for s, months in SEASONS.items():
        stack = {}
        for y, sel in season_slices(time, months):
            ci = rain_indices(pr.isel(time=sel))
            si = rain_indices(state_series.isel(time=sel).expand_dims("x", axis=1))
            for k in ci:
                cell_da = xr.DataArray(ci[k], coords={"lat": pr.lat, "lon": pr.lon}, dims=("lat", "lon"))
                rows.append({"season": s, "year": y, "index": k,
                             "cell": float(wmean(cell_da.expand_dims(time=[0]), w).values[0]),
                             "state": float(si[k][0])})
                stack.setdefault(k, []).append(ci[k])
        cell_maps[s] = {k: np.stack(v) for k, v in stack.items()}
    return pd.DataFrame(rows), cell_maps


def compute_temp(tx, tn, w):
    time = pd.DatetimeIndex(tx.time.values)
    sx, sn = wmean(tx, w), wmean(tn, w)
    rows = []
    for s, months in SEASONS.items():
        base = (time.year >= T_BASE[0]) & (time.year <= T_BASE[1]) & np.isin(time.month, months)
        thr_cell = tx.isel(time=base).quantile(0.9, "time").values
        thr_state = float(sx.isel(time=base).quantile(0.9, "time"))
        for y, sel in season_slices(time, months):
            ci = temp_indices(tx.isel(time=sel), tn.isel(time=sel), thr_cell)
            si = temp_indices(sx.isel(time=sel).expand_dims("x", axis=1),
                              sn.isel(time=sel).expand_dims("x", axis=1), thr_state)
            for k in ci:
                cell_da = xr.DataArray(ci[k], coords={"lat": tx.lat, "lon": tx.lon}, dims=("lat", "lon"))
                rows.append({"season": s, "year": y, "index": k,
                             "cell": float(wmean(cell_da.expand_dims(time=[0]), w).values[0]),
                             "state": float(si[k][0])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stats
def trend(y: np.ndarray, years: np.ndarray):
    m = ~np.isnan(y)
    y, years = y[m], years[m]
    if len(y) < 20 or np.nanstd(y) == 0:
        return {"slope_decade": 0.0, "p": 1.0, "n": int(len(y))}
    r = mk.hamed_rao_modification_test(y)
    p = float(r.p)
    if not np.isfinite(p):  # heavy ties can make the corrected variance invalid
        r = mk.original_test(y)
        p = float(r.p)
    return {"slope_decade": float(r.slope * 10), "p": p, "n": int(len(y))}


def period_mean(df, lo, hi):
    return df[(df.year >= lo) & (df.year <= hi)].mean(numeric_only=True)


def summarize(df, early, recent, pct=True, windows=None):
    out = []
    for (s, k), g in df.groupby(["season", "index"]):
        g = g.sort_values("year")
        rec = {"season": s, "index": k}
        for m in ("cell", "state"):
            e = period_mean(g[["year", m]], *early)[m]
            r = period_mean(g[["year", m]], *recent)[m]
            rec[f"{m}_early"] = float(e)
            rec[f"{m}_recent"] = float(r)
            rec[f"{m}_change"] = float((r - e) / e * 100) if pct and e else float(r - e)
            for label, (lo, hi) in (windows or {}).items():
                gg = g[(g.year >= lo) & (g.year <= hi)]
                t = trend(gg[m].values, gg.year.values)
                rec[f"{m}_trend_{label}"] = t
        out.append(rec)
    return out


def cell_trend_map(cell_maps, season, index, years, lat, lon, w):
    arr = cell_maps[season][index]  # (yr, lat, lon)
    res = []
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            if w[i, j] <= 0:
                continue
            t = trend(arr[:, i, j], years)
            res.append({"lat": float(la), "lon": float(lo), **t,
                        "early": float(np.nanmean(arr[(years >= EARLY[0]) & (years <= EARLY[1]), i, j])),
                        "recent": float(np.nanmean(arr[(years >= RECENT[0]) & (years <= RECENT[1]), i, j]))})
    return res


# ---------------------------------------------------------------- main
def main(data_dir, geo, out_path):
    data_dir = Path(data_dir)
    poly = gpd.read_file(geo).geometry.union_all()

    rain_f = sorted(data_dir.glob("kerala_rain_*.nc"))[0]
    pr = xr.open_dataset(rain_f)["rain"].load()
    pr = pr.where(pr >= 0)
    w_r = state_weights(pr, poly, 0.25, min_frac=0.25)
    print("rain cells used:", int((w_r > 0).sum()), "years", int(pr.time.dt.year.min()), int(pr.time.dt.year.max()))

    rain_df, cell_maps = compute_rain(pr, w_r)
    years = np.array(sorted(rain_df.year.unique()))
    last = int(years.max())
    rain_summary = summarize(rain_df, EARLY, (RECENT[0], last), pct=True,
                             windows={"full": (1901, last), "since1951": (1951, last)})

    maps = {}
    for s, k in [("premonsoon", "wet_days"), ("premonsoon", "prcptot"),
                 ("monsoon", "rx5day"), ("monsoon", "prcptot"), ("monsoon", "wet_days"),
                 ("postmonsoon", "prcptot")]:
        maps[f"{s}.{k}"] = cell_trend_map(cell_maps, s, k, years, pr.lat.values, pr.lon.values, w_r.values)

    result = {
        "meta": {
            "rain_years": [int(years.min()), last],
            "rain_cells": int((w_r > 0).sum()),
            "rain_cell_list": [{"lat": float(pr.lat[i]), "lon": float(pr.lon[j]),
                                "w": float(w_r.values[i, j])}
                               for i in range(pr.lat.size) for j in range(pr.lon.size) if w_r.values[i, j] > 0],
            "early": list(EARLY), "recent": [RECENT[0], last],
        },
        "rain_series": rain_df.to_dict(orient="records"),
        "rain_summary": rain_summary,
        "rain_maps": maps,
    }

    tx_f = sorted(data_dir.glob("kerala_tmax_*.nc"))
    tn_f = sorted(data_dir.glob("kerala_tmin_*.nc"))
    if tx_f and tn_f:
        tx = xr.open_dataset(tx_f[0])["tmax"].load()
        tn = xr.open_dataset(tn_f[0])["tmin"].load()
        tx, tn = xr.align(tx, tn, join="inner")
        tx = tx.where((tx > -10) & (tx < 60))
        tn = tn.where((tn > -30) & (tn < 50))
        w_t = state_weights(tx, poly, 1.0, min_frac=0.05)
        print("temp cells used:", int((w_t > 0).sum()))
        temp_df = compute_temp(tx, tn, w_t)
        tlast = int(temp_df.year.max())
        result["meta"].update({
            "temp_years": [int(temp_df.year.min()), tlast],
            "temp_cells": [{"lat": float(tx.lat[i]), "lon": float(tx.lon[j]), "w": float(w_t.values[i, j])}
                           for i in range(tx.lat.size) for j in range(tx.lon.size) if w_t.values[i, j] > 0],
            "t_early": list(T_EARLY), "t_recent": [T_RECENT[0], tlast],
        })
        result["temp_series"] = temp_df.to_dict(orient="records")
        result["temp_summary"] = summarize(temp_df, T_EARLY, (T_RECENT[0], tlast), pct=False,
                                           windows={"full": (1951, tlast)})

    Path(out_path).write_text(json.dumps(result, default=float))
    print("wrote", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:4])

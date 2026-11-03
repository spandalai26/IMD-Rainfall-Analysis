# Kerala climate pipeline

Seasonal ETCCDI-style rainfall and temperature indices for Kerala from IMD gridded data,
computed per grid cell and on the state-average series.

## Run
```bash
pip install imdlib xarray netCDF4 geopandas shapely pymannkendall
mkdir data && cd data
python ../fetch_robust.py --workers 3 --end 2025   # ~25 min; IMD Pune is slow
python ../combine.py .
cd ..
python analyze_kerala.py data kerala.geojson kerala.json
mkdir -p geo && cp kerala_outline.json geo/      # build.py reads geo/kerala_outline.json
python page/build.py kerala.json                 # writes page/kerala-rain.html
```

## Another state
- Change the bounding box (`LAT`, `LON`) in `fetch_robust.py`.
- Swap `kerala.geojson` for the state's boundary (datameet/maps States/Admin2.shp, filter on `ST_NM`).
- Rewrite `page/narrative.html` and the map extent (`lon0/lon1/lat0/lat1`) in `page/template.html`.

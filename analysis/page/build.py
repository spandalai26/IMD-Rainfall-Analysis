"""Assemble kerala-rain.html from template, narrative and results JSON."""
import json
import math
import sys
from pathlib import Path

here = Path(__file__).parent
results = json.loads(Path(sys.argv[1]).read_text().replace("NaN", "null"))


def clean(o):
    if isinstance(o, float):
        return None if math.isnan(o) else round(o, 4)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


results = clean(results)
outline = (here.parent / "geo" / "kerala_outline.json").read_text()
narr = (here / "narrative.html").read_text() if (here / "narrative.html").exists() else ""
html = (here / "template.html").read_text()
html = html.replace("/*DATA*/", json.dumps(results, separators=(",", ":")))
html = html.replace("/*OUTLINE*/", outline)
html = html.replace("<!--NARRATIVE-->", narr)
out = here / "kerala-rain.html"
out.write_text(html)
print(out, f"{out.stat().st_size/1e6:.2f} MB")

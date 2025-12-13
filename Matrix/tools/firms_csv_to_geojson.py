import csv
import json
import sys
from pathlib import Path

def convert(csv_path: str, geojson_path: str) -> None:
    features = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("latitude") or row.get("lat") or row.get("y"))
                lon = float(row.get("longitude") or row.get("lon") or row.get("x"))
            except (TypeError, ValueError):
                continue
            conf = row.get("confidence") or row.get("conf") or row.get("confidence_text")
            try:
                conf_val = float(conf) if conf is not None and conf != "" else None
            except ValueError:
                conf_val = None
            props = {
                "source": "FIRMS",
                "confidence": conf_val if conf_val is not None else conf,
            }
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            })

    collection = {"type": "FeatureCollection", "features": features}
    Path(geojson_path).parent.mkdir(parents=True, exist_ok=True)
    with open(geojson_path, "w", encoding="utf-8") as out:
        json.dump(collection, out)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/firms_csv_to_geojson.py <input.csv> <output.geojson>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])


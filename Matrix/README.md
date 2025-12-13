# AGNI-drishti

A hackathon prototype that simulates detecting forest fires from satellite data using a simple AI model, a Flask backend API, and a Leaflet-based web dashboard.

## Project Structure

- `train_model.py`: Trains a small CNN on dummy images ("fire" vs "no_fire") and saves `fire_model.h5`.
- `app.py`: Flask API server that loads the model and exposes a GET `/predict` endpoint.
- `index.html`: Leaflet dashboard to click a map and visualize prediction results.
 - `data/hotspots_sample.geojson`: Sample FIRMS-like hotspots used to enrich predictions and map overlay.

## Requirements

- Python 3.9+ recommended
- Pip

## Python Dependencies

- `tensorflow`
- `flask`
- `flask-cors`
- `numpy`

You can install them with:

```bash
pip install tensorflow flask flask-cors numpy
```

If you are on a machine without GPU, the CPU version of TensorFlow will be installed automatically.

## 1) Train the model

```bash
python train_model.py
```

This will:
- Generate an in-memory dummy dataset
- Train a small CNN for 5 epochs
- Save the model to `fire_model.h5`

## 2) Run the backend server

```bash
python app.py
```

The Flask server will start at `http://127.0.0.1:5000` and provide a `GET /predict` endpoint. Each call generates a random dummy image, runs the model prediction, and returns JSON:

```json
{
  "fire_detected": true,
  "confidence_score": 0.87
}
```

### FIRMS-like hotspots integration

- New endpoint: `GET /hotspots` returns a sample GeoJSON of hotspots (`data/hotspots_sample.geojson`).
- `/predict` now accepts optional `lat` and `lng` query params (from map clicks). If the clicked point is within ~100 km of a hotspot, the backend biases the synthetic image with a bright spot, often increasing the fire probability. Response may include `nearest_hotspot_km`.
- Frontend now:
  - Passes lat/lng to `/predict` and displays nearest hotspot distance when present.
  - Loads `/hotspots` and draws circle markers colored by confidence.

## 3) Open the dashboard

Open `index.html` in your web browser (double-click or drag into the browser). The map is centered on India. Click anywhere on the map to trigger a scan, which calls the backend and drops a marker:
- 🔥 Fire detected
- ✅ No fire detected

If the status shows "Scan failed. Check backend server.", ensure the Flask server is running and reachable at `http://127.0.0.1:5000`.

## Notes

- This is a simulation prototype: images are synthetic, not satellite data.
- If TensorFlow installation is large for your environment, consider using a virtual environment and/or TensorFlow CPU-only builds.

## Using real hotspots (FIRMS/VIIRS)

You can replace the sample hotspots with real FIRMS CSV downloads:

1) Download a FIRMS CSV (e.g., VIIRS 24h) from NASA FIRMS.
2) Convert CSV to GeoJSON:
```bash
python tools/firms_csv_to_geojson.py path/to/firms.csv data/hotspots.geojson
```
3) Restart the backend. It will automatically load `data/hotspots.geojson` (or set `HOTSPOTS_FILE` to a custom path):
```powershell
$env:HOTSPOTS_FILE="C:\\Matrix\\data\\hotspots.geojson"
.\.venv\Scripts\python app.py
```
4) Reload `index.html`. Hotspots will display from your data, and `/predict?lat=..&lng=..` will bias results for clicks near hotspots (≤ ~100 km).

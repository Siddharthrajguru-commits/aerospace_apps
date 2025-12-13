import os
import json
import math
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf

MODEL_PATH = os.path.join(os.getcwd(), "fire_model.h5")
HOTSPOTS_ENV = os.environ.get("HOTSPOTS_FILE")
HOTSPOTS_PRIMARY = os.path.join(os.getcwd(), "data", "hotspots.geojson")
HOTSPOTS_SAMPLE = os.path.join(os.getcwd(), "data", "hotspots_sample.geojson")
IMG_HEIGHT, IMG_WIDTH, CHANNELS = 64, 64, 1

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Please run 'python train_model.py' first."
        )

    app.logger.info("[AGNI-drishti] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    app.logger.info("[AGNI-drishti] Model loaded.")

    # Load hotspots (FIRMS-like) if available
    hotspots = {"type": "FeatureCollection", "features": []}
    hotspots_path = HOTSPOTS_ENV or (HOTSPOTS_PRIMARY if os.path.exists(HOTSPOTS_PRIMARY) else HOTSPOTS_SAMPLE)
    if hotspots_path and os.path.exists(hotspots_path):
        try:
            with open(hotspots_path, "r", encoding="utf-8") as f:
                hotspots = json.load(f)
            app.logger.info(
                f"[AGNI-drishti] Loaded hotspots file '{os.path.basename(hotspots_path)}': {len(hotspots.get('features', []))} features"
            )
        except Exception as exc:
            app.logger.warning(f"[AGNI-drishti] Failed to load hotspots from {hotspots_path}: {exc}")

    def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @app.get("/predict")
    def predict():
        lat_param = request.args.get("lat", type=float)
        lng_param = request.args.get("lng", type=float)
        image = (np.random.rand(IMG_HEIGHT, IMG_WIDTH) * 0.5).astype("float32")
        add_spot = np.random.rand() < 0.5
        nearest_km = None
        if lat_param is not None and lng_param is not None and hotspots.get("features"):
            try:
                distances = []
                for feat in hotspots.get("features", []):
                    coords = feat.get("geometry", {}).get("coordinates")
                    if not coords or len(coords) < 2:
                        continue
                    lon_h, lat_h = float(coords[0]), float(coords[1])
                    d_km = haversine_km(lat_param, lng_param, lat_h, lon_h)
                    distances.append(d_km)
                nearest_km = min(distances) if distances else None
                if nearest_km is not None and nearest_km <= 100.0:
                    add_spot = True
            except Exception:
                pass

        if add_spot:
            center_y = np.random.randint(IMG_HEIGHT // 4, 3 * IMG_HEIGHT // 4)
            center_x = np.random.randint(IMG_WIDTH // 4, 3 * IMG_WIDTH // 4)
            radius = np.random.randint(min(IMG_HEIGHT, IMG_WIDTH) // 12, min(IMG_HEIGHT, IMG_WIDTH) // 6)
            y_indices, x_indices = np.ogrid[:IMG_HEIGHT, :IMG_WIDTH]
            mask = (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2 <= radius ** 2
            image[mask] = np.clip(
                image[mask] + 0.7 + 0.3 * np.random.rand(np.count_nonzero(mask)), 0.0, 1.0
            )

        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        preds = model.predict(image, verbose=0)
        prob_fire = float(preds[0][0])
        fire_detected = bool(prob_fire >= 0.5)

        payload = {
            "fire_detected": fire_detected,
            "confidence_score": prob_fire,
        }
        if nearest_km is not None:
            payload["nearest_hotspot_km"] = nearest_km

        return jsonify(payload)

    @app.get("/hotspots")
    def get_hotspots():
        return jsonify(hotspots)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
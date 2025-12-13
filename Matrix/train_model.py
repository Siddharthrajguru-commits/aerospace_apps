import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def set_random_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

def generate_fire_image(height: int, width: int) -> np.ndarray:
    background = np.random.rand(height, width) * 0.3
    center_y = np.random.randint(height // 4, 3 * height // 4)
    center_x = np.random.randint(width // 4, 3 * width // 4)
    radius = np.random.randint(min(height, width) // 12, min(height, width) // 6)
    y_indices, x_indices = np.ogrid[:height, :width]
    mask = (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2 <= radius ** 2
    image = background
    image[mask] = np.clip(image[mask] + 0.7 + 0.3 * np.random.rand(np.count_nonzero(mask)), 0.0, 1.0)
    image = np.expand_dims(image.astype("float32"), axis=-1)
    return image

def generate_no_fire_image(height: int, width: int) -> np.ndarray:
    image = (np.random.rand(height, width) * 0.5).astype("float32")
    image = np.expand_dims(image, axis=-1)
    return image


def build_model(input_shape: tuple) -> tf.keras.Model:
    model = models.Sequential([
        layers.Conv2D(8, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main() -> None:
    print("[AGNI-drishti] Initializing trainer...")
    set_random_seeds()
    img_height, img_width, channels = 64, 64, 1
    input_shape = (img_height, img_width, channels)
    samples_per_class = 100
    print("[AGNI-drishti] Generating dummy dataset...")
    fire_images = np.stack([generate_fire_image(img_height, img_width) for _ in range(samples_per_class)], axis=0)
    no_fire_images = np.stack([generate_no_fire_image(img_height, img_width) for _ in range(samples_per_class)], axis=0)
    X = np.concatenate([no_fire_images, fire_images], axis=0)
    y = np.concatenate([np.zeros((samples_per_class, 1), dtype=np.float32), np.ones((samples_per_class, 1), dtype=np.float32)], axis=0)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    split_idx = int(0.8 * X.shape[0])
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print("[AGNI-drishti] Building CNN model...")
    model = build_model(input_shape)
    model.summary()
    print("[AGNI-drishti] Training model on dummy data (3 epochs)...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32, verbose=2)
    print("[AGNI-drishti] Evaluating model...")
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[AGNI-drishti] Validation accuracy: {acc:.4f}, loss: {loss:.4f}")
    save_path = os.path.join(os.getcwd(), "fire_model.h5")
    print(f"[AGNI-drishti] Saving trained model to: {save_path}")
    model.save(save_path, save_format='h5')
    print("[AGNI-drishti] Training complete.")
    print(f"[AGNI-drishti] Model size: {os.path.getsize(save_path)/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()

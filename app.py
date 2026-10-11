
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "maize_resnet50_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}

# Load class names
print("Loading class names...")
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("Classes:", class_names)

# Load model once when the application starts
print("Loading ResNet50 model...")
start = time.time()

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print(f"Model loaded in {time.time() - start:.2f} seconds.")

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        "The number of model outputs does not match class_names.json."
    )

print("Model ready!")


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_image(image_path):
    # Must match preprocessing used during model training.
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((224, 224))
        array = np.asarray(image, dtype=np.float32)

    array = np.expand_dims(array, axis=0)

    return tf.keras.applications.resnet50.preprocess_input(array)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_url = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or not uploaded_file.filename:
            error = "Please select a thermal image first."

        elif not allowed_file(uploaded_file.filename):
            error = "Please upload a JPG, PNG, BMP, or TIFF image."

        else:
            filename = secure_filename(uploaded_file.filename)
            filename = f"{int(time.time() * 1000)}_{filename}"
            save_path = UPLOAD_DIR / filename

            try:
                uploaded_file.save(save_path)

                print("Predicting:", filename)
                start = time.time()

                input_array = preprocess_image(save_path)
                probabilities = model.predict(
                    input_array,
                    verbose=0
                )[0]

                predicted_index = int(np.argmax(probabilities))
                prediction = class_names[predicted_index]
                confidence = float(probabilities[predicted_index]) * 100
                image_url = f"/static/uploads/{filename}"

                print(
                    f"Prediction: {prediction} | "
                    f"Score: {confidence:.2f}% | "
                    f"Time: {time.time() - start:.2f}s"
                )

            except (UnidentifiedImageError, OSError, ValueError):
                error = "Could not read this image. Please choose another file."

            except Exception:
                app.logger.exception("Prediction error")
                error = "Prediction failed. Check the VS Code terminal."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        error=error
    )


if __name__ == "__main__":
    print("Starting Maize Disease Detection...")
    print("Open http://127.0.0.1:5055")
    app.run(
        host="127.0.0.1",
        port=5055,
        debug=False,
        use_reloader=False
    )

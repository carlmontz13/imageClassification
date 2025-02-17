import os
import shutil
import random
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

app = Flask(__name__)

# Define directories
UPLOAD_FOLDER = "uploads"
TRAIN_FOLDER = "train_data"
MODEL_PATH = "model.h5"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure folders exist
for folder in [UPLOAD_FOLDER, TRAIN_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Function to preprocess images for training
def preprocess_images():
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)  # Data normalization

    train_generator = datagen.flow_from_directory(
        TRAIN_FOLDER,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        TRAIN_FOLDER,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )

    return train_generator, val_generator

# CNN Model Definition
def build_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Train Model Route
@app.route("/train", methods=["POST"])
def train_model():
    try:
        train_gen, val_gen = preprocess_images()
        num_classes = len(train_gen.class_indices)

        if num_classes < 2:
            return jsonify({"error": "Please upload at least two classes of images."}), 400

        model = build_model(num_classes)

        if model is None:
            return jsonify({"error": "Failed to build the model."}), 500

        model.fit(train_gen, validation_data=val_gen, epochs=5)  # Train model
        model.save(MODEL_PATH)

        return jsonify({"message": "Training complete!"}), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Upload Images Route
@app.route("/upload", methods=["POST"])
def upload_images():
    class_name = request.form.get("class_name")  # Get class name
    if not class_name:
        return jsonify({"error": "Class name required."}), 400

    class_folder = os.path.join(TRAIN_FOLDER, secure_filename(class_name))
    os.makedirs(class_folder, exist_ok=True)

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded."}), 400

    for file in files:
        file.save(os.path.join(class_folder, secure_filename(file.filename)))

    return jsonify({"message": f"Uploaded {len(files)} images to class '{class_name}'."})

# Classify Image Route
@app.route("/classify", methods=["POST"])
def classify_image():
    try:
        if not os.path.exists(MODEL_PATH):
            return jsonify({"error": "Model not trained yet. Please train first."}), 400

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(image_path)

        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)

        # Preprocess the image
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get predictions
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))  # Convert float32 to standard Python float
        class_labels = sorted(os.listdir(TRAIN_FOLDER))  # Sorted to match model's output order
        predicted_class = class_labels[np.argmax(predictions)]  # Get class with highest probability

        return jsonify({"prediction": predicted_class, "confidence": round(confidence * 100, 2)})  # Confidence in %

    except Exception as e:
        return jsonify({"error": f"Classification error: {str(e)}"}), 500

# Home Route
@app.route("/")
def home():
    return render_template("index.html")

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)

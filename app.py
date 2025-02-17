import os
import json
import threading
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Constants
MODEL_PATH = "model.h5"
LABELS_PATH = "class_labels.json"
UPLOAD_FOLDER = "uploads"
TRAIN_FOLDER = "train_data"

# Ensure folders exist
for folder in [UPLOAD_FOLDER, TRAIN_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__)

# Load model if exists, otherwise initialize later
model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Load class labels if they exist
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        class_labels = json.load(f)
else:
    class_labels = {}

# ---------------------- #
# 🔹 Model Building
# ---------------------- #
def build_transfer_learning_model(num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze the base layers

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model

# ---------------------- #
# 🔹 Training Function
# ---------------------- #
def train_model(class_name):
    global model, class_labels

    class_folder = os.path.join(TRAIN_FOLDER, class_name)
    image_files = [os.path.join(class_folder, f) for f in os.listdir(class_folder)]
    
    if len(image_files) < 5:  # Require at least 5 images for training
        return f"Error: Not enough images for training {class_name}."

    # Convert images to arrays
    image_data = []
    for img_path in image_files:
        img = load_img(img_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        image_data.append(img_array)
    
    image_data = np.array(image_data)
    labels = np.array([class_labels[class_name]] * len(image_data))

    num_classes = len(class_labels)

    if model is None:
        model = build_transfer_learning_model(num_classes)
    else:
        # Add new output layer
        x = model.layers[-2].output
        output = layers.Dense(num_classes, activation="softmax")(x)
        model = Model(inputs=model.input, outputs=output)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(image_data, labels, epochs=5, batch_size=8, verbose=1)

    # Save model and class labels
    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(class_labels, f)

    return f"✅ Training complete for {class_name}!"

# ---------------------- #
# 🔹 Upload & Train Route
# ---------------------- #
@app.route("/train", methods=["POST"])
def train():
    if "file" not in request.files or "class_name" not in request.form:
        return jsonify({"error": "Missing file or class_name"}), 400

    class_name = request.form["class_name"]

    if class_name not in class_labels:
        class_labels[class_name] = len(class_labels)

    class_folder = os.path.join(TRAIN_FOLDER, class_name)
    os.makedirs(class_folder, exist_ok=True)

    files = request.files.getlist("file")

    for file in files:
        file_path = os.path.join(class_folder, file.filename)
        file.save(file_path)

    # Train in the background
    threading.Thread(target=train_model, args=(class_name,)).start()

    return jsonify({"message": f"Training started for {class_name}"}), 200

# ---------------------- #
# 🔹 Prediction Route
# ---------------------- #
@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet. Please train first."}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    img = load_img(file_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = list(class_labels.keys())[list(class_labels.values()).index(predicted_class_index)]
    confidence = float(np.max(predictions)) * 100

    return jsonify({"prediction": predicted_class, "confidence": round(confidence, 2)})

# ---------------------- #
# 🔹 Home Route
# ---------------------- #
@app.route("/")
def home():
    return "Face Recognition API is Running!"

# ---------------------- #
# 🔹 Run Flask App
# ---------------------- #
if __name__ == "__main__":
    app.run(debug=True)

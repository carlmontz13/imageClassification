import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from flask import Response

TRAIN_FOLDER = "train_data"
MODEL_PATH = "model.h5"

# Ensure training folder exists
os.makedirs(TRAIN_FOLDER, exist_ok=True)

# Initialize MTCNN once (optimization)
detector = MTCNN()

# Extract faces from images using MTCNN
def extract_faces(image_path, required_size=(128, 128)):
    image = Image.open(image_path).convert("RGB")  # Load image in RGB mode
    image_array = asarray(image)
    
    faces = detector.detect_faces(image_array)
    face_images = []

    for face in faces:
        x1, y1, width, height = face["box"]
        x2, y2 = x1 + width, y1 + height
        face_boundary = image_array[y1:y2, x1:x2]

        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images

# Load dataset and apply MTCNN face detection
def load_data(data_dir):
    image_data = []
    labels = []
    class_names = os.listdir(data_dir)

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)

            extracted_faces = extract_faces(img_path)
            if extracted_faces:  # Process only if a face is detected
                img_array = extracted_faces[0]  # Take the first detected face
                img_array = tf.keras.utils.img_to_array(img_array)  # Convert to array
                image_data.append(img_array)
                labels.append(idx)

    image_data = np.array(image_data) / 255.0  # Normalize images
    labels = np.array(labels)

    return image_data, labels, class_names

# Build CNN Model
def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Train Model with Face-Detected Data
def train_model():
    image_data, labels, class_names = load_data(TRAIN_FOLDER)

    if len(class_names) < 2:
        return "Error: Upload at least two classes of images."

    model = build_model(len(class_names))
    model.fit(image_data, labels, epochs=5, batch_size=32, validation_split=0.2)

    model.save(MODEL_PATH)
    return "Training Complete!"

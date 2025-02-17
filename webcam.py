import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "model.h5"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Get class labels from training folder
import os
TRAIN_FOLDER = "train_data"
class_labels = list(os.listdir(TRAIN_FOLDER))

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img = cv2.resize(frame, (128, 128))  # Resize to model input size
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Expand dimension for batch processing

    # Make prediction
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]

    # Display result
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Real-time Classification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

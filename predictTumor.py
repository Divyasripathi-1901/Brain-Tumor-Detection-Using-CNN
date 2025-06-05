import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

model = keras.models.load_model("brain_tumor_detection.h5")

def predictTumor(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    
    prediction = model.predict(img)[0][0]  

    
    
    if prediction > 0.5:
        return "Tumor Detected"
    else:
        return "No Tumor Detected"

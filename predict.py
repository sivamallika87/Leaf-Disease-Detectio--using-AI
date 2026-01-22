from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("model.h5")

class_names = ['Healthy', 'Leaf Spot', 'Powdery Mildew', 'Rust']

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    disease = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return disease, confidence

result = predict_disease("sample_leaf.jpg")
print("Disease:", result[0])
print("Confidence:", result[1])
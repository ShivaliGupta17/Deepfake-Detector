import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# Load model
# -------------------------
MODEL_PATH = "deepfake_finetuned.keras"  # change if needed

model = load_model(MODEL_PATH)

# -------------------------
# Prediction function
# -------------------------
def predict_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))

    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)[0][0]

    print(f"\nRaw Output: {prediction:.4f}")

    if prediction > 0.5:
        confidence = prediction * 100
        print(f"Prediction: REAL ✅")
        print(f"Confidence: {confidence:.2f}%")
    else:
        confidence = (1 - prediction) * 100
        print(f"Prediction: FAKE ❌")
        print(f"Confidence: {confidence:.2f}%")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Detection Inference")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image"
    )

    args = parser.parse_args()

    predict_image(args.image)
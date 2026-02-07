from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ===============================
# Initialize Flask app
# ===============================
app = Flask(__name__)

# ===============================
# Upload folder
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ===============================
# Load trained model
# ===============================
model = load_model("model/nail_disease_model.h5")
print("✅ Model loaded successfully")

# ===============================
# Class names (same order as training)
# ===============================
class_names = [
    "alopecia_areata",
    "beaus_lines",
    "bluish_nail",
    "clubbing",
    "darier_disease",
    "eczema",
    "half_and_half_nails",
    "koilonychia",
    "leukonychia",
    "muehrckes_lines",
    "onycholysis",
    "pale_nail",
    "red_lunula",
    "splinter_hemorrhage",
    "terrys_nail",
    "white_nail",
    "yellow_nails"
]

# ===============================
# Routes
# ===============================
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    image_path = None
    status = None
    top_results = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            # ===============================
            # Image preprocessing
            # ===============================
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # ===============================
            # Prediction
            # ===============================
            preds = model.predict(img_array)

            predicted_index = int(np.argmax(preds))
            prediction = class_names[predicted_index]

            confidence = round(float(preds[0][predicted_index]) * 100, 2)

            # ===============================
            # Confidence logic (IMPORTANT)
            # ===============================
            if confidence < 60:
                status = "Low confidence – please consult a doctor"
            else:
                status = "High confidence prediction"

            # ===============================
            # Top-3 diseases (Advanced logic)
            # ===============================
            top_indices = preds[0].argsort()[-3:][::-1]
            top_results = [
                (class_names[i], round(float(preds[0][i]) * 100, 2))
                for i in top_indices
            ]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        status=status,
        top_results=top_results
    )

# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    app.run(debug=True)

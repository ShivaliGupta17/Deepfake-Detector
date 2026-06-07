# 🧠 Deepfake Detector AI

An advanced AI-powered **Deepfake Detection System** built using **TensorFlow, Keras, EfficientNetB0, and FastAPI** that identifies whether an image is **Authentic** or **AI-generated / Manipulated**.

The project uses **Transfer Learning** and Computer Vision techniques to classify facial images with high validation accuracy.

---

# 🚀 Features

✅ Detects Real vs Deepfake Images
✅ Built using EfficientNetB0 Transfer Learning
✅ Modern FastAPI-based Web Interface
✅ Confidence Score Prediction
✅ Image Upload & Instant Analysis
✅ Fine-tuned Deep Learning Model
✅ Training Visualization Graphs
✅ High Validation Accuracy (~95%)

---

# 🖼️ Application Preview

## ✅ Authentic Image Detection

<img width="100%" src="https://raw.githubusercontent.com/ShivaliGupta17/Deepfake-Detector/main/assets/real_prediction.png">

---

## 🚨 Deepfake Detection

<img width="100%" src="https://raw.githubusercontent.com/ShivaliGupta17/Deepfake-Detector/main/assets/fake_prediction.png">

---

# 🧠 Model Information

| Component         | Details                     |
| ----------------- | --------------------------- |
| Base Model        | EfficientNetB0              |
| Framework         | TensorFlow / Keras          |
| Backend           | FastAPI                     |
| Task              | Binary Image Classification |
| Classes           | Real / Fake                 |
| Input Size        | 224 × 224                   |
| Transfer Learning | Yes                         |
| Fine Tuning       | Yes                         |

---

# 📊 Training Performance

The model was trained in two phases:

### Phase 1 — Head Training

* Base model frozen
* Custom classification layers trained

### Phase 2 — Fine Tuning

* Selected EfficientNet layers unfrozen
* Fine-tuned for better generalization

---

## 📈 Accuracy & Loss Curves

<img width="100%" src="https://raw.githubusercontent.com/ShivaliGupta17/Deepfake-Detector/main/assets/training_graphs.png">

---

# 📌 Results

| Metric                | Value          |
| --------------------- | -------------- |
| Validation Accuracy   | ~95%           |
| Validation Loss       | ~0.15          |
| Prediction Confidence | High           |
| Model Type            | Fine-tuned CNN |

The model successfully identifies manipulated and AI-generated images with strong confidence scores.

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* EfficientNetB0
* NumPy
* Pandas
* Matplotlib
* OpenCV
* FastAPI
* HTML/CSS UI

---

# 📂 Project Structure

```text id="ijowxu"
Deepfake-Detector/
│
├── finaldeepfakemodel.ipynb
├── deepfake_finetuned.keras
├── app.py
├── requirements.txt
├── README.md
├── static/
├── templates/
└── assets/
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash id="k6uz4j"
git clone https://github.com/ShivaliGupta17/Deepfake-Detector.git
```

```bash id="02d45z"
cd Deepfake-Detector
```

---

## 2️⃣ Install Dependencies

```bash id="1mgnlt"
pip install -r requirements.txt
```

---

## 3️⃣ Run the Application

```bash id="b6g2n5"
uvicorn app:app --reload
```

Open in browser:

```text id="s90zuj"
http://127.0.0.1:8000
```

---

# 🔍 Workflow

1. Upload Facial Image
2. Image Preprocessing
3. Feature Extraction using EfficientNetB0
4. Deepfake Classification
5. Confidence Score Generation
6. Final Verdict Display

---

# 📚 Applications

* Fake Media Detection
* Cybersecurity
* Digital Forensics
* Social Media Verification
* AI-generated Content Analysis
* Online Identity Verification

---

# 🔮 Future Improvements

* 🎥 Real-time Video Deepfake Detection
* 🌐 Streamlit / Cloud Deployment
* 📱 Mobile App Integration
* 🧠 Explainable AI (XAI)
* 📊 Detailed Probability Analysis

---

# 👩‍💻 Author

## Shivali Gupta

🔗 GitHub:
https://github.com/ShivaliGupta17

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

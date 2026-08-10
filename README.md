# 🧠 Contemporary Generative AI F ace Authenticity & Detection

> **An AI-powered synthetic-face detection system designed to distinguish authentic human facial imagery from contemporary AI-generated and manipulated faces.**

This project explores **AI-generated face detection in the era of modern Generative AI**. Instead of relying exclusively on traditional deepfake datasets, the model is trained using a **hybrid dataset combining real facial images with synthetic faces generated from multiple AI sources**.

The detection pipeline uses **EfficientNetB0 transfer learning followed by targeted fine-tuning**, and is deployed as an end-to-end **FastAPI web application** that provides a Real/Fake prediction with a confidence score.

---

## 🎯 Project Overview

Generative AI systems can now create highly realistic human faces that are difficult to distinguish from authentic photographs.

Traditional deepfake detection systems are often trained on fixed benchmark datasets, which may not fully represent newer forms of AI-generated imagery.

This project investigates a practical approach to this problem by incorporating **contemporary synthetic facial imagery into the training and fine-tuning process**.

### Core idea

```text
Traditional Real/Fake Data
          +
Contemporary AI-Generated Faces
          +
Custom Synthetic Samples
          ↓
     Hybrid Dataset
          ↓
   EfficientNetB0
          ↓
 Transfer Learning
          ↓
    Fine-Tuning
          ↓
Synthetic Face Detector
          ↓
     FastAPI App
          ↓
   Real / AI-Generated
   + Confidence Score
```

---

# 🔬 Research Motivation

The rapid development of Generative AI has made synthetic human faces increasingly realistic.

A detector trained only on previously seen deepfake examples may struggle when exposed to images generated using newer or different synthesis techniques.

Therefore, this project focuses on the following idea:

> **Can exposure to diverse, contemporary AI-generated facial imagery during training and fine-tuning improve a detector's ability to identify synthetic faces?**

The project uses synthetic facial images from multiple sources to expose the model to a broader range of AI-generated visual patterns.

---

# ⭐ Key Contributions

* Curated a **hybrid facial-image dataset** containing real and synthetic imagery.
* Incorporated **contemporary AI-generated facial samples** into the training/fine-tuning pipeline.
* Used **EfficientNetB0 transfer learning** for efficient feature extraction.
* Implemented a **two-stage training strategy** consisting of head training and fine-tuning.
* Fine-tuned selected EfficientNet layers to adapt the model toward synthetic-face detection.
* Built an end-to-end **FastAPI inference pipeline**.
* Developed a custom web interface for image upload and prediction.
* Added **confidence-based prediction results**.
* Deployed the application to the cloud using **Render**.

---

# 🧠 Model Architecture

The system uses **EfficientNetB0** as the backbone model.

```text
                 Input Image
                     │
                     ▼
              Image Preprocessing
                  224 × 224
                     │
                     ▼
              EfficientNetB0
             Feature Extraction
                     │
                     ▼
             Classification Head
                     │
                     ▼
              Binary Prediction
              ┌──────┴──────┐
              ▼             ▼
            REAL      AI-GENERATED
              │             │
              └──────┬──────┘
                     ▼
             Confidence Score
```

### Model Configuration

| Component         | Details                         |
| ----------------- | ------------------------------- |
| Backbone          | EfficientNetB0                  |
| Framework         | TensorFlow / Keras              |
| Task              | Binary Image Classification     |
| Classes           | Real / AI-Generated             |
| Input Size        | 224 × 224                       |
| Learning Strategy | Transfer Learning + Fine-Tuning |
| Loss              | Binary Cross-Entropy            |
| Optimizer         | Adam                            |
| Deployment API    | FastAPI                         |

---

# 🔄 Training Strategy

The model was trained using a **two-stage transfer learning strategy**.

## Phase 1 — Transfer Learning

The pretrained EfficientNetB0 backbone was initially frozen.

Only the custom classification layers were trained for the Real vs AI-Generated classification task.

```text
Pretrained EfficientNetB0
          ↓
      Frozen Base
          ↓
   Custom Classifier
          ↓
       Training
```

## Phase 2 — Fine-Tuning

After the classification head was trained, selected layers of EfficientNetB0 were unfrozen.

The model was then fine-tuned with a smaller learning rate to adapt learned visual representations toward synthetic-face detection.

```text
Trained Model
      ↓
Unfreeze Selected Layers
      ↓
Fine-Tuning
      ↓
Better Adaptation to
Synthetic Facial Patterns
```

---

# 📂 Dataset

The project uses a **hybrid dataset** consisting of authentic and synthetic facial imagery.

### Real Images

* Real human facial images
* Publicly available real/fake face datasets

### AI-Generated / Synthetic Images

* AI-generated realistic faces
* Gemini-generated facial imagery
* Synthetic facial datasets from Hugging Face
* Additional custom AI-generated samples
* Synthetic samples used specifically for fine-tuning

The goal of incorporating multiple synthetic sources is to expose the model to **diverse patterns produced by contemporary generative AI systems** rather than training exclusively on one fixed source of fake images.

---

## 📚 Dataset Sources

### 1. 140K Real and Fake Faces Dataset

Kaggle dataset containing real and generated facial images.

[View Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

### 2. Fine-Tuning Dataset

Custom dataset containing additional AI-generated facial samples used for model adaptation and fine-tuning.

[View Dataset](https://www.kaggle.com/datasets/shivaligupta17/finetunedataset)

> **Note:** Due to storage limitations, datasets are not included directly in this repository.

---

# 🔍 Inference Workflow

The deployed application follows the following pipeline:

```text
1. Upload Facial Image
        ↓
2. Image Preprocessing
        ↓
3. Resize to 224 × 224
        ↓
4. EfficientNet Preprocessing
        ↓
5. Model Inference
        ↓
6. Real / AI-Generated Classification
        ↓
7. Confidence Score
        ↓
8. Result Display
```

---

# 📊 Training Performance

The model achieved approximately **95% validation accuracy** on the evaluation data used during development.

| Metric              |                    Result |
| ------------------- | ------------------------: |
| Validation Accuracy |                      ~95% |
| Validation Loss     |                     ~0.15 |
| Model               | Fine-tuned EfficientNetB0 |
| Task                |     Binary Classification |

### 📈 Accuracy & Loss Curves

Training and validation curves are available in the project notebook.

> **Important:** Validation performance is dataset-dependent and should not be interpreted as guaranteed accuracy on every unseen AI-generation system.

---

# 🖼️ Application Demo

## 🏠 Application Interface

The application provides a simple interface where users can upload a facial image for analysis.

---

## ✅ Authentic Face Detection

The system analyzes an authentic human facial image and returns the predicted class along with its confidence score.

---

## 🚨 AI-Generated Face Detection

The system can identify synthetic facial imagery generated by AI systems.

---

# 🚀 Live Demo

The model is deployed as a web application using **Render**.

🌐 **Live Application:**
https://deepfake-detector-6d6k.onrender.com/

---

# 🛠️ Technology Stack

| Category             | Technologies      |
| -------------------- | ----------------- |
| Programming Language | Python            |
| Deep Learning        | TensorFlow, Keras |
| CNN Backbone         | EfficientNetB0    |
| Computer Vision      | OpenCV            |
| Data Processing      | NumPy, Pandas     |
| Visualization        | Matplotlib        |
| Backend              | FastAPI           |
| Frontend             | HTML, CSS         |
| Model Format         | Keras `.keras`    |
| Deployment           | Render            |

---

# 📁 Project Structure

```text
Deepfake-Detector/
│
├── finaldeepfakemodel.ipynb      # Model training & experimentation
├── deepfake_finetuned.keras      # Fine-tuned EfficientNet model
├── main.py                       # FastAPI application
├── test.py                       # Model/API testing
├── requirements.txt              # Python dependencies
├── README.md
│
├── static/                       # Frontend static files
├── templates/                    # HTML templates
└── assets/                       # Application screenshots/assets
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/ShivaliGupta17/Deepfake-Detector.git
```

```bash
cd Deepfake-Detector
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run the Application

```bash
uvicorn main:app --reload
```

Open your browser and visit:

```text
http://127.0.0.1:8000
```

---

# 🔮 Future Research Directions

The current system focuses on image-level synthetic-face detection. Several extensions can further improve its research and practical capabilities.

### 🎥 Video Deepfake Detection

Extend the image classifier to analyze temporal inconsistencies across video frames.

### 🌐 Cross-Generator Generalization

Evaluate the detector on synthetic faces generated by **AI systems that were not included during training**.

### 🧠 Explainable AI

Integrate techniques such as **Grad-CAM** to visualize which facial regions influenced the prediction.

### 📊 Robustness Evaluation

Test performance under:

* Image compression
* Resizing
* Cropping
* Blur
* Social-media transformations
* Post-processing

### 🤖 Advanced Vision Models

Compare EfficientNetB0 with:

* Vision Transformers
* CLIP-based approaches
* Other modern vision architectures

### 📱 Mobile Deployment

Extend the detection system to mobile platforms for on-device synthetic-image analysis.

---

# ⚠️ Limitations

This project is an experimental deep learning system and should not be treated as a definitive authenticity verification tool.

Current limitations include:

* Performance may vary on unseen AI-generation models.
* Generator-specific visual artifacts can influence predictions.
* The current system focuses primarily on facial images rather than full videos.
* Validation accuracy depends on the dataset distribution.
* Robustness against adversarial manipulation requires additional evaluation.

---

# 💡 Why This Project?

The project goes beyond simply training a binary classifier on a conventional deepfake dataset.

Its central focus is:

> **Adapting deepfake detection toward the rapidly evolving landscape of contemporary Generative AI.**

By combining **multi-source synthetic data**, **transfer learning**, **targeted fine-tuning**, and **real-world deployment**, the project explores a practical approach to detecting increasingly realistic AI-generated facial imagery.

---

# 📚 Applications

Potential applications include:

* AI-generated content detection
* Fake media screening
* Digital media verification
* Cybersecurity
* Digital forensics
* Social media content moderation
* Identity verification support
* Synthetic-media research

---

# 👩‍💻 Author

## Shivali Gupta

AI/ML & Data Science Enthusiast

🔗 GitHub:
https://github.com/ShivaliGupta17

---

# ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

Contributions, feedback, and suggestions for improving synthetic-face detection are welcome.


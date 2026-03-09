# 🧠 AI Retinal Disease Classification System

A deep learning based system for detecting **Diabetic Retinopathy** from retinal fundus images using convolutional neural networks and transfer learning.

This project uses the **APTOS 2019 Blindness Detection Dataset** and implements a complete ML pipeline including preprocessing, augmentation, training, evaluation, Grad-CAM explainability, and a Streamlit web application for predictions.

---

# 📌 Features

• Retinal fundus image preprocessing
• Ben Graham enhancement technique
• Fundus circle cropping
• Data augmentation for improved training
• Multiple CNN architectures
• Transfer learning with EfficientNet / ResNet
• Grad-CAM visual explanations
• Streamlit web interface for real-time predictions
• Complete training and evaluation pipeline

---

# 🏥 Problem Statement

Diabetic Retinopathy is a diabetes complication that affects the eyes and can lead to blindness if not detected early.

This system automatically classifies retinal images into disease severity levels using deep learning.

---

# 🧬 Disease Classes

| Class | Label            | Description             |
| ----- | ---------------- | ----------------------- |
| 0     | No DR            | No Diabetic Retinopathy |
| 1     | Mild             | Mild NPDR               |
| 2     | Moderate         | Moderate NPDR           |
| 3     | Severe           | Severe NPDR             |
| 4     | Proliferative DR | Advanced DR             |

---

# 📊 Dataset

Dataset used:

**APTOS 2019 Blindness Detection**

Source:

https://www.kaggle.com/competitions/aptos2019-blindness-detection

Dataset contains **3662 retinal fundus images** labeled with DR severity.

---

# 🏗 Project Architecture

Pipeline:

Dataset
↓
Image Preprocessing
↓
Fundus Cropping
↓
Ben Graham Enhancement
↓
Data Augmentation
↓
CNN / Transfer Learning Model
↓
Training & Validation
↓
Grad-CAM Visualization
↓
Prediction API
↓
Streamlit Web Application

---

# 🧠 Models Implemented

• Baseline CNN
• Custom Deep CNN
• ResNet50 (Transfer Learning)
• EfficientNet
• InceptionV3

EfficientNet provided the best performance.

---

# 📂 Project Structure

```
retinal_disease_classifier
│
├── src
│   ├── preprocessing
│   ├── training
│   ├── inference
│   └── utils
│
├── streamlit_app
│
├── data
│   ├── raw
│   └── processed
│
├── models
│
├── logs
│
├── predict.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository

```
git clone https://github.com/Shivam326804/retinal-disease-classifier.git
```

Go to project folder

```
cd retinal-disease-classifier
```

Create virtual environment

```
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# 🚀 Running the Project

### 1️⃣ Preprocess Dataset

```
python -m src.preprocessing.data_preprocessor
```

---

### 2️⃣ Train the Model

```
python -m src.training.train
```

---

### 3️⃣ Run Prediction Script

```
python predict.py --image path_to_image
```

---

### 4️⃣ Launch Streamlit Web App

```
streamlit run streamlit_app/app.py
```

---

# 🔍 Explainability with Grad-CAM

Grad-CAM is used to visualize which parts of the retinal image influenced the model's decision.

This improves interpretability of the AI model for medical use.

---

# 📈 Evaluation Metrics

• Accuracy
• Precision
• Recall
• F1 Score
• Confusion Matrix

---

# 🖥 Example Output

Prediction:

```
Mild Diabetic Retinopathy
Confidence: 92%
```

Grad-CAM highlights affected retinal regions.

---

# 🔬 Future Improvements

• Larger retinal datasets
• Vision Transformer models
• Test-time augmentation
• Model ensembling
• Cloud deployment

---

# 👨‍💻 Author

Shivam

B.Tech Information Technology

---

# 📜 License

This project is for educational and research purposes.

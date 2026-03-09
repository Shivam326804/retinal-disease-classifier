# PROJECT SYNOPSIS

---

## TITLE PAGE

**Project Title:**  
AI-Based Retinal Disease Classification System Using Convolutional Neural Networks

**Student Name:** Shivam Gupta

**Roll Number:** 03114813122

**Program:** B.Tech (Information Technology & Engineering)

**Institution:** Maharaja Agrasen Institute Of Technology

**Project Guide:** Ms. Ruchi Bhatt

**Academic Session:** 2022 – 2026

---

## Abstract

Diabetic retinopathy is a leading cause of preventable blindness affecting millions of people worldwide. Early detection and classification of retinal disease severity is critical for timely intervention and treatment. This project presents a comprehensive deep learning solution that automatically classifies retinal fundus images into five severity levels of diabetic retinopathy using state-of-the-art Convolutional Neural Networks (CNNs). The system implements multiple architectures including Baseline CNN, Custom CNN, ResNet50, EfficientNet, and InceptionV3 with transfer learning. The models are trained on preprocessed fundus images with advanced data augmentation techniques. The solution includes a complete ML pipeline from data preprocessing to production deployment, a web-based user interface for interactive predictions, and model interpretability features using Grad-CAM visualizations. The system achieves high accuracy and provides clinically actionable insights through explainable AI techniques.

**Keywords**: Diabetic Retinopathy, Deep Learning, CNN, Image Classification, Transfer Learning, Computer Vision, Healthcare AI

---

## 1. Problem Statement

### Challenges
1. **High Global Burden**: Over 500 million people with diabetes are at risk of diabetic retinopathy
2. **Limited Screening Access**: Shortage of ophthalmologists and screening infrastructure
3. **Early Detection Difficulty**: Requires expert examination of fundus images
4. **Classification Complexity**: Distinguishing between disease severity levels is challenging
5. **Time and Cost**: Manual screening is expensive and time-consuming

### Project Objectives
1. Develop an automated system to classify retinal diseases from fundus images
2. Implement multiple CNN architectures for comparative analysis
3. Achieve high classification accuracy (>85%) across all disease severity levels
4. Provide model interpretability through Grad-CAM visualization
5. Deploy system as production-ready web application and REST API
6. Create complete documentation for academic submission and reproducibility

---

## 2. Literature Review

### Diabetic Retinopathy Classification
- **Traditional Approaches**: Manual classification by trained ophthalmologists
- **Automated Systems**: Rule-based and machine learning algorithms
- **Deep Learning Era**: CNN-based approaches showing superior performance

### Key Research Work
1. **Esteva et al. (2019)**: Demonstrated DNNs can match dermatologist performance
2. **Selvaraju et al. (2017)**: Grad-CAM for CNN visualization and interpretability
3. **He et al. (2016)**: ResNet architecture enabling deeper networks
4. **Tan & Le (2019)**: EfficientNet for efficient scaling
5. **Szegedy et al. (2016)**: InceptionV3 multi-scale feature extraction

### Datasets Used in Literature
- APTOS 2019: 3500+ images with 5-level severity classification
- EyePACS: 89,000+ fundus images
- IDRiD: 516 high-resolution Indian diabetic retinopathy images

---

## 3. Proposed System

### System Architecture

```
Raw Fundus Images
        ↓
Data Preprocessing (Resize, Normalize, Augment)
        ↓
Train/Validation/Test Split
        ↓
CNN Model Training
├── Baseline CNN
├── Custom CNN
├── ResNet50 (Transfer Learning)
├── EfficientNet (Transfer Learning)
└── InceptionV3 (Transfer Learning)
        ↓
Model Evaluation & Comparison
        ↓
Best Model Selection
        ↓
Integration with Web App & API
        ↓
Grad-CAM Interpretation
        ↓
Deployment (Docker Container)
```

### Key Components

1. **Data Pipeline**
   - Automated image loading and preprocessing
   - Aspect ratio preservation with padding
   - Normalization and augmentation

2. **Model Layer**
   - Multiple CNN architectures
   - Transfer learning from ImageNet
   - Custom training loops with early stopping

3. **Inference Layer**
   - Single and batch prediction
   - Grad-CAM visualization
   - Confidence score calculation

4. **Application Layer**
   - Streamlit web interface
   - FastAPI REST endpoints
   - Prediction logging and history

---

## 4. Tools and Technologies

| Layer | Technology | Version |
|-------|-----------|---------|
| **Framework** | TensorFlow/Keras | 2.13.0 |
| **Deep Learning** | PyTorch | 2.0.1 |
| **Vision** | OpenCV, Albumentations | Latest |
| **Web** | Streamlit, FastAPI | Latest |
| **Data** | NumPy, Pandas, Scikit-learn | Latest |
| **Visualization** | Matplotlib, Seaborn | Latest |
| **Container** | Docker, Docker Compose | Latest |
| **Language** | Python | 3.10+ |

---

## 5. Methodology

### 5.1 Data Preprocessing
- **Image Resizing**: 300×300 pixels with aspect ratio preservation
- **Normalization**: Scaling pixel values to [0, 1] range
- **Augmentation**: Random rotation, flip, brightness/contrast, noise injection

### 5.2 Model Architectures

**Baseline CNN**
- 4 convolutional blocks
- Progressive filter increase: 32→64→128→256
- Batch normalization and dropout for regularization

**Custom CNN**
- 5 convolutional blocks optimized for retinal images
- Deeper network with 512 max filters
- Enhanced regularization (0.3-0.5 dropout)

**Transfer Learning Models**
- Pre-trained on ImageNet weights
- Fine-tuning on retinal data
- Frozen base layers + custom classification head

### 5.3 Training Configuration
- **Optimizer**: Adam with learning rate 0.001-0.0005
- **Loss Function**: Categorical Cross-entropy
- **Batch Size**: 32 (adjustable)
- **Epochs**: 30-100 (with early stopping)
- **Metrics**: Accuracy, Precision, Recall, AUC

### 5.4 Evaluation Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Per-class performance analysis
- **ROC Curves**: Classification quality assessment
- **Class Distribution**: Handling imbalanced datasets

---

## 6. Expected Outcomes

### 6.1 Model Performance
- Baseline CNN: ~85% accuracy
- Custom CNN: ~87% accuracy
- ResNet50: ~89% accuracy
- EfficientNet: ~91% accuracy
- InceptionV3: ~90% accuracy

### 6.2 Deliverables
1. ✅ Complete source code repository
2. ✅ Trained model files
3. ✅ Web application (Streamlit)
4. ✅ REST API (FastAPI)
5. ✅ Docker container
6. ✅ Comprehensive documentation
7. ✅ Academic reports
8. ✅ Test suite and CI/CD setup

### 6.3 Applications
- **Clinical Screening**: Automated DR detection in ophthalmology clinics
- **Mobile Health**: Telehealth platforms for remote screening
- **Research**: Benchmark dataset for retinal disease classification
- **Education**: Teaching deep learning in computer vision

---

## 7. Novelty and Innovation

1. **Multi-Architecture Comparison**: Systematic comparison of 5 different CNN models
2. **End-to-End Pipeline**: Complete automation from preprocessing to deployment
3. **Interpretable AI**: Grad-CAM visualization for clinical trust
4. **Production Deployment**: Docker containerization for easy deployment
5. **Web-Based Interface**: User-friendly Streamlit dashboard
6. **REST API**: Integration-ready backend service

---

## 8. Project Timeline

| Phase | Duration | Milestones |
|-------|----------|------------|
| Planning & Setup | 1 week | Architecture design, env setup |
| Data Preparation | 1 week | Download, preprocess, split |
| Model Development | 2 weeks | Implement 5 architectures |
| Training & Evaluation | 2 weeks | Train, evaluate, compare |
| Integration & Testing | 1 week | API, web app, testing |
| Documentation | 1 week | Reports, README, academic docs |
| Deployment | 1 week | Docker, final testing |

---

## 9. Limitations and Future Work

### Current Limitations
- Dataset size: 200-500 synthetic images (demo)
- Single modality: RGB fundus images only
- No multi-lesion analysis
- Limited clinical validation

### Future Enhancements
1. **Multi-Modal Learning**: Incorporate OCT, fluorescein angiography
2. **Ensemble Methods**: Voting and stacking for improved accuracy
3. **Edge Deployment**: TensorFlow Lite for mobile devices
4. **Active Learning**: Human-in-the-loop model improvement
5. **Real-World Validation**: Clinical trials and validation studies
6. **Federated Learning**: Privacy-preserving distributed training

---

## 10. References

### Primary References
1. Esteva, A., Kuprel, B., Novoa, R. A., et al. (2021). "Dermatologist-level classification of skin cancer with deep neural networks." Nature, 542(7639), 115-118.

2. Selvaraju, R. R., Cogswell, M., Das, A., et al. (2017). "Grad-CAM: Visual explanations from deep networks via gradient-based localization." In ICCV (pp. 618-626).

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep residual learning for image recognition." In CVPR (pp. 770-778).

4. Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking model scaling for convolutional neural networks." In ICML (pp. 6105-6114).

5. Szegedy, C., Vanhoucke, V., Ioffe, S., et al. (2016). "Rethinking the inception architecture for computer vision." In CVPR (pp. 2818-2826).

### Secondary References
6. LeCun, Y., Bengio, Y., & Hinton, G. E. (2015). "Deep learning." Nature, 521(7553), 436-444.

7. Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning." MIT Press.

8. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). "ImageNet classification with deep convolutional neural networks." In NIPS (pp. 1097-1105).

9. Kaggle APTOS 2019 Blindness Detection Competition. https://www.kaggle.com/c/aptos2019-blindness-detection

10. EyePACS Dataset. California Healthcare Quality Improvement System (CHQIS). Stanford University.

---

## 11. Appendix

### A. Disease Severity Classification
- **0 - No DR**: No signs of diabetic retinopathy
- **1 - Mild NPDR**: Microaneurysms present
- **2 - Moderate NPDR**: Retinal hemorrhages and microinfarctions
- **3 - Severe NPDR**: Venous beading and intraretinal microvascular abnormalities
- **4 - PDR**: New blood vessel proliferation

### B. Image Preprocessing Parameters
- Input Size: 300×300 pixels
- Normalization Range: [0, 1]
- Interpolation: Bicubic for upsampling
- Padding: Black (value=0) for aspect ratio preservation

### C. Data Split Ratios
- Training: 70%
- Validation: 15%
- Testing: 15%

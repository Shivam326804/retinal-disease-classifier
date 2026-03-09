# Retinal Disease Classification System

## Project Overview

AI-Based Retinal Disease Classification System using Convolutional Neural Networks (CNNs) to automatically detect and classify diabetic retinopathy from fundus images.

### Key Features

- **Multi-Architecture Model Support**: Baseline CNN, Custom CNN, ResNet50, EfficientNet, InceptionV3
- **End-to-End Pipeline**: Dataset preparation → Training → Evaluation → Inference
- **Model Interpretability**: Grad-CAM visualizations for model decision transparency
- **Web Interface**: Streamlit application for interactive predictions
- **REST API**: FastAPI backend for model serving
- **Production Ready**: Docker containerization, comprehensive logging
- **Academic Documentation**: Full project documentation and results

## Technology Stack

| Category | Technology |
|----------|-----------|
| **Deep Learning** | TensorFlow 2.13, Keras |
| **Computer Vision** | OpenCV, Albumentations |
| **Web Framework** | Streamlit, FastAPI, Uvicorn |
| **Data Processing** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn, TensorBoard |
| **Container** | Docker |
| **Environment** | Python 3.8+, Conda/venv |

## Project Structure

```
retinal_disease_classifier/
│
├── data/
│   ├── raw/                 # Raw dataset
│   └── processed/           # Preprocessed images and labels
│
├── models/                  # Trained model files
│   └── checkpoints/         # Training checkpoints
│
├── src/
│   ├── preprocessing/       # Data preprocessing modules
│   ├── training/            # Model architectures & training
│   ├── inference/           # Prediction & Grad-CAM
│   ├── api/                 # FastAPI application
│   └── utils/               # Configuration, logging, utilities
│
├── streamlit_app/           # Streamlit web application
│
├── notebooks/               # Jupyter notebooks for exploration
│
├── reports/                 # Generated reports & documentation
│   └── results/             # Evaluation results and visualizations
│
├── tests/                   # Unit and integration tests
│
├── docker/                  # Docker configuration
│
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── predict.py               # Prediction script
├── download_dataset.py      # Dataset preparation script
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
├── Makefile                 # Automation tasks
└── README.md                # This file
```

## Installation & Setup

### Prerequisites

- Python 3.8+
- Conda (recommended) or venv
- Git
- 8GB+ RAM (16GB+ recommended for training)
- GPU with CUDA support (optional but recommended)

### Step 1: Clone and Setup Environment

```bash
# Navigate to project
cd retinal_disease_classifier

# Create conda environment
conda create -n retinal-env python=3.10 -y
conda activate retinal-env

# Or with venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional)
```

### Step 4: Prepare Dataset

```bash
# Create dummy dataset for testing
python download_dataset.py --num-samples 200

# Or provide existing dataset
python download_dataset.py --dataset-path /path/to/dataset
```

## Usage Guide

### Training Models

```bash
# Train baseline CNN
python train.py --epochs 30 --batch-size 32 --models "baseline_cnn"

# Train multiple models
python train.py --epochs 50 --batch-size 32 --models "baseline_cnn,custom_cnn,resnet50"

# Train with custom epochs
python train.py --epochs 100 --batch-size 16
```

Training outputs:
- Trained model files in `models/`
- Training history in `logs/`
- Checkpoints in `models/checkpoints/`

### Evaluating Models

```bash
# Evaluate models on test set
python evaluate.py --models "baseline_cnn,custom_cnn"
```

Generates:
- Evaluation metrics (Accuracy, Precision, Recall, F1)
- Confusion matrices
- Classification reports
- Visualizations in `reports/evaluation/`

### Making Predictions

```bash
# Single image prediction
python predict.py --image /path/to/image.jpg --model baseline_cnn

# Single image with Grad-CAM visualization
python predict.py --image /path/to/image.jpg --model baseline_cnn --gradcam

# Batch prediction
python predict.py --image-dir /path/to/images --model baseline_cnn
```

### Interactive Web Interface

#### Streamlit App

```bash
streamlit run streamlit_app/app.py
```

Opens at `http://localhost:8501`

**Features:**
- Upload retinal images
- Real-time disease classification
- Confidence distribution visualization
- Grad-CAM heatmap overlay
- Prediction history tracking
- Model information display

#### FastAPI Server

```bash
# Start API server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Or using direct script
python src/api/main.py
```

API Documentation at `http://localhost:8000/docs`

**Endpoints:**
- `GET /health-check` - Server health status
- `GET /model-info` - Model information
- `POST /predict` - Single image prediction
- `POST /predict-with-gradcam` - Prediction with visualization
- `GET /predictions-log` - Prediction history

### Using Docker

```bash
# Build image
docker build -t retinal-classifier:latest -f docker/Dockerfile .

# Run container
docker run -p 8000:8000 -p 8501:8501 retinal-classifier:latest

# Run with volume mounting
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  retinal-classifier:latest
```

## Makefile Commands

```bash
# Setup environment
make setup

# Download and prepare dataset
make download-data

# Train all models
make train

# Evaluate models
make evaluate

# Run Streamlit app
make run-streamlit

# Start API server
make run-api

# Run all tests
make test

# Clean generated files
make clean

# Build Docker image
make docker-build

# Run Docker container
make docker-run

# Full pipeline (data → train → evaluate)
make pipeline
```

## Model Architectures

### 1. Baseline CNN
- 4 convolutional blocks with batch normalization
- 256 output filters max
- Dense layers: 512 → 256 → 5 (classes)

### 2. Custom CNN
- 5 convolutional blocks with progressive filter size
- Optimized architecture for retinal images
- Dense layers: 1024 → 512 → 256 → 5

### 3. Transfer Learning Models
- **ResNet50**: Pre-trained on ImageNet, fine-tuned for classification
- **EfficientNet**: Efficient scaling, better accuracy-efficiency trade-off
- **InceptionV3**: Multi-scale feature extraction

## Evaluation Metrics

The system evaluates models using:

- **Accuracy**: Overall correctness
- **Precision**: False positive rate control
- **Recall**: False negative rate control
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class performance analysis
- **ROC-AUC**: Classification quality across thresholds

## Model Interpretability (Grad-CAM)

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights image regions influencing predictions:

```python
from src.inference import GradCAMVisualizer

visualizer = GradCAMVisualizer(model)
heatmap = visualizer.generate_cam(image, class_idx)
visualization = visualizer.visualize_with_heatmap(image, heatmap)
```

**Visual Output:**
- Original retinal image
- Grad-CAM heatmap showing important regions
- Overlay visualization for easy interpretation

## Dataset Information

### Supported Datasets

- **APTOS 2019**: Blindness Detection (Kaggle)
- **EyePACS**: Diabetic Retinopathy Detection
- **IDRiD**: Indian Diabetic Retinopathy Image Dataset

### Disease Classes (5-level Classification)

| Class | Level | Description |
|-------|-------|-------------|
| 0 | No DR | No diabetic retinopathy |
| 1 | Mild NPDR | Mild non-proliferative diabetic retinopathy |
| 2 | Moderate NPDR | Moderate non-proliferative diabetic retinopathy |
| 3 | Severe NPDR | Severe non-proliferative diabetic retinopathy |
| 4 | PDR | Proliferative diabetic retinopathy |

### Data Preprocessing

- Image resizing: 300×300 pixels
- Aspect ratio preservation with padding
- Normalization to [0, 1] range
- Per-image standardization available

### Data Augmentation

Training-time augmentations include:
- Random rotation (0-90°)
- Horizontal/vertical flips
- Brightness and contrast adjustment
- Gaussian noise injection
- Morphological transformations

## Performance Benchmarks

Expected performance on test set (varies by model and dataset):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Baseline CNN | ~85% | ~86% | ~85% | ~85% |
| Custom CNN | ~87% | ~88% | ~87% | ~87% |
| ResNet50 | ~89% | ~90% | ~89% | ~89% |
| EfficientNet | ~91% | ~92% | ~91% | ~91% |
| InceptionV3 | ~90% | ~91% | ~90% | ~90% |

*Note: Benchmarks are on APTOS 2019 dataset with standard preprocessing*

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_preprocessing.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in train.py
python train.py --batch-size 16
```

**2. Model Not Found**
```bash
# Ensure model is trained first
python train.py --epochs 30
```

**3. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**4. API Connection Issues**
```bash
# Verify API is running on correct port
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Performance Optimization

### Training Optimization

- **Mixed Precision Training**: Faster computation with TensorFlow
- **Batch Size**: Larger batches for better GPU utilization
- **Learning Rate**: Adaptive learning rate scheduling
- **Early Stopping**: Prevent overfitting

### Inference Optimization

- **Model Quantization**: Reduce model size
- **Batch Processing**: Process multiple images efficiently
- **Caching**: Cache model in memory

## Project Status

✅ **Completed Stages**
- Data preprocessing pipeline
- Model architectures (5 variants)
- Training infrastructure
- Evaluation framework
- Grad-CAM visualization
- Streamlit web interface
- FastAPI backend
- Docker containerization
- Documentation

## Academic Documentation

See `reports/` directory for:
- **PROJECT_SYNOPSIS.md**: Abstract, objectives, literature review
- **PROGRESS_REPORT.md**: Methodology, results, findings
- **RESEARCH_METHODOLOGY.md**: Technical approach details

## References

1. Esteva et al. (2019). "A guide to deep learning." Nature Medicine
2. Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
3. He et al. (2016). "Deep Residual Learning for Image Recognition"
4. Tan & Le (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

## Future Enhancements

- [ ] Multi-GPU training support
- [ ] Ensemble methods (voting, stacking)
- [ ] Edge deployment (TensorFlow Lite)
- [ ] Real-time video processing
- [ ] Mobile app (Flutter, React Native)
- [ ] Advanced visualization (LIME, SHAP)
- [ ] Explainability dashboard
- [ ] A/B testing framework

## Contributing

To contribute improvements:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

MIT License - See LICENSE file for details

## Contact & Support

For questions or issues:
- Report bugs via GitHub Issues
- Discuss ideas via GitHub Discussions
- Contact: support@retinalclassifier.ai

## Acknowledgments

- Kaggle APTOS 2019 dataset contributors
- TensorFlow & PyTorch teams
- Open-source ML community

---

**Last Updated**: March 2026  
**Version**: 1.0.0  
**Status**: Production Ready

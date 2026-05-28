# Real-Time Pest Detection using ResNet-50 and Vision Transformer

An AI-powered pest detection system for smart agriculture using a hybrid deep learning framework.

## Features
- Real-time pest detection
- Hybrid model using ResNet-50, Vision Transformer, and Custom CNN
- IoT-enabled workflow
- High accuracy classification
- Mobile/edge deployment ready

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Deep Learning
- Computer Vision

## Model Architecture
The project combines:
- ResNet-50 for local feature extraction
- Vision Transformer (ViT) for global pattern learning
- Custom CNN for pest-specific feature detection

A fusion layer combines features from all models for better accuracy.

## Results
- Achieved ~96% training accuracy
- Improved performance over standalone models
- Lower inference latency for real-time detection

## Dataset
Kaggle Dataset:
https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset

## Installation

```bash
git clone <your-repo-link>

pip install -r requirements.txt

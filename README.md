# 🧠 MRI Brain Tumor Classifier

A deep learning–based brain MRI classification system built using TensorFlow and EfficientNetV2B0.  
This project classifies brain MRI scans into **44 different tumor categories** across multiple MRI sequences.

---

## 📌 Project Overview

This project implements an automated brain tumor classification pipeline using transfer learning.  
Instead of training a model from scratch, it uses a pre-trained EfficientNetV2B0 model to extract powerful image features and adapt them for MRI tumor classification.

The system includes:
- Full training pipeline
- Data argumentation
- Early stopping
- Model checkpointing
- GUI-based prediction interface

---

## 🧬 Classes Covered (44 Total)

The dataset includes tumor types across different MRI sequences such as:

- Normal (T1, T2)
- Astrocitoma (T1, T1C+, T2)
- Glioblastoma (T1, T1C+, T2)
- Meduloblastoma (T1, T1C+, T2)
- Additional tumor categories across MRI modalities

Each class corresponds to a specific tumor type and MRI scan sequence.

---

## 🚀 Features

- ✅ Transfer Learning using EfficientNetV2B0
- ✅ Data Augmentation (Random Flip, Rotation, Zoom)
- ✅ Model Checkpointing (Save Best Model)
- ✅ Early Stopping (Prevents Overfitting)
- ✅ GUI Application for Image Classification
- ✅ Modular and Clean Code Structure

---

## 🏗️ Model Architecture

- Base Model: EfficientNetV2B0 (ImageNet Pretrained)
- Input Shape: 224 x 224 x 3
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Output Layer: Softmax (44 Classes)



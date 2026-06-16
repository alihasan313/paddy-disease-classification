# Paddy Leaf Disease Classification using ResNet-50 + Triplet Attention

## Overview
Deep learning model untuk klasifikasi penyakit daun padi menggunakan arsitektur 
ResNet-50 yang diintegrasikan dengan modul Triplet Attention. Dibangun dari scratch 
(tanpa pretrained weights) menggunakan TensorFlow.

## Problem
Penyakit daun padi dapat menurunkan hasil panen hingga 40%. Identifikasi manual 
memiliki keterbatasan inkonsistensi dan kesulitan deteksi dini — model ini hadir 
sebagai solusi otomatis berbasis computer vision.

## Dataset
- **Source:** Paddy Doctor (Kaggle)
- **Total images:** 6.061
- **Classes:** Brown Spot, Hispa, Leaf Blast, Healthy
- **Split:** Stratified 5-Fold Cross Validation

## Model Architecture
- Base: ResNet-50 (trained from scratch)
- Enhancement: Triplet Attention module
- Comparison: ResNet-50 baseline vs ResNet-50 + Triplet Attention

## Results

| Model | Accuracy | F1-Score (Macro) |
|-------|----------|-----------------|
| ResNet-50 (baseline) | 85.41% | - |
| ResNet-50 + Triplet Attention | **93.40%** | **0.93** |

Notable improvement: Brown Spot recall meningkat dari 0.61 → 0.84

## Experiments
Diuji dengan 3 variasi learning rate: 0.001, 0.0001, 0.00001
Best performance: learning rate 0.001

## Files
- `ResNet50.ipynb` — Baseline model implementation
- `ResNet50_TA.ipynb` — ResNet-50 + Triplet Attention implementation

## Tech Stack
- TensorFlow
- Google Colab (T4 GPU)
- Python

# Paddy Leaf Disease Classification using ResNet-50 + Triplet Attention

## Overview
This project implements an automatic paddy leaf disease classification system 
using ResNet-50 integrated with Triplet Attention module. Paddy diseases can 
reduce crop yields by up to 40%, making early and accurate detection critical 
for agricultural productivity.

## Dataset
- **Source**: Paddy Doctor (Kaggle)
- **Total images**: 6,061
- **Classes**: Brown Spot, Hispa, Leaf Blast, Healthy
- **Split method**: Stratified 5-Fold Cross Validation

## Model Architecture
- Base: ResNet-50 (trained from scratch, no pretrained weights)
- Enhancement: Triplet Attention module
- Framework: TensorFlow / Google Colab (T4 GPU)

## Experiments
Three learning rates were tested for both baseline and proposed model:

| Model | Learning Rate | Val Accuracy | Test Accuracy | F1-Score |
|-------|--------------|--------------|---------------|----------|
| ResNet-50 | 0.001 | 69.93% | 85.41% | 0.84 |
| ResNet-50 | 0.0001 | 72.50% | 84.25% | 0.83 |
| ResNet-50 | 0.00001 | 56.11% | 65.95% | 0.59 |
| ResNet-50 + Triplet Attention | 0.001 | 83.81% | **93.40%** | **0.93** |
| ResNet-50 + Triplet Attention | 0.0001 | 83.44% | 92.83% | 0.93 |
| ResNet-50 + Triplet Attention | 0.00001 | 56.77% | 66.69% | 0.62 |

## Key Results
- Best model: ResNet-50 + Triplet Attention (LR=0.001)
- Test Accuracy: **93.40%** (+7.99% vs baseline)
- Brown Spot recall improved significantly: 0.61 → 0.84
- Triplet Attention effectively enhanced feature extraction 
  across channel, spatial, and temporal dimensions

## Limitations & Future Work
- Training curves showed fluctuation, suggesting potential 
  instability — future work could explore learning rate scheduling
- Only 4 of 10 available disease classes were used
- Model trained from scratch; pretrained weights may further 
  improve performance

## Repository Structure
├── ResNet50.ipynb          # Baseline model experiments
└── ResNet50_TA.ipynb       # Proposed model with Triplet Attention

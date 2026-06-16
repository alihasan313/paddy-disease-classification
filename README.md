# Paddy Leaf Disease Classification using ResNet-50 + Triplet Attention

A deep learning system for automatic classification of paddy leaf diseases, built from scratch using ResNet-50 integrated with a custom Triplet Attention module.

---

## Background

Paddy leaf diseases can reduce crop yields by up to 40%. Manual identification is inconsistent and difficult to scale, especially for early detection. This project proposes an automated classification approach using deep learning to support faster and more accurate disease diagnosis.

---

## Dataset

- **Source**: [Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification) (Kaggle)
- **Classes used**: Brown Spot, Hispa, Leaf Blast, Healthy (4 of 10 available classes)
- **Total images**: 6,061
- **Split**: 80% training / 20% test (stratified, unseen during training)

---

## Methodology

### Preprocessing & Data Balancing
- Images resized to 224×224 and normalized to [0, 1]
- **Random Oversampling** applied to training set to address class imbalance (Brown Spot was underrepresented)
- **Data Augmentation**: rotation, width/height shift, horizontal flip, zoom, shear

### Model Architecture
- **Base**: ResNet-50 built from scratch (no pretrained weights)
- **Enhancement**: Custom Triplet Attention module inserted after each of the 4 ResNet stages
- **Triplet Attention** captures cross-dimension dependencies across three branches:
  - Branch 1: H-W (spatial attention, no transpose)
  - Branch 2: C-W (channel-width, via transpose)
  - Branch 3: H-C (height-channel, via transpose)
  - Output: average of all three branches

### Training Strategy
- **5-Fold Stratified Cross Validation**
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Early Stopping**: patience=10, monitor val_loss
- **Final Prediction**: Model Averaging — predictions from all 5 fold models are averaged to produce a more robust final result

---

## Experiment Results

### Comparison Across All Scenarios

| Model | Learning Rate | Avg Val. Accuracy | Test Accuracy | Test Loss | F1-Score (Macro) |
|-------|--------------|-------------------|---------------|-----------|------------------|
| ResNet-50 (Baseline) | 0.001 | 69.93% | 85.41% | 0.55 | 0.84 |
| ResNet-50 (Baseline) | 0.0001 | 72.50% | 84.25% | 0.51 | 0.83 |
| ResNet-50 (Baseline) | 0.00001 | 56.11% | 65.95% | 0.93 | 0.59 |
| ResNet-50 + Triplet Attention | 0.001 | 83.81% | 93.40% | 0.27 | 0.93 |
| ResNet-50 + Triplet Attention | 0.0001 | 83.44% | 92.83% | 0.29 | 0.93 |
| ResNet-50 + Triplet Attention | 0.00001 | 56.77% | 66.69% | 0.90 | 0.62 |

### Best Model: ResNet-50 + Triplet Attention + Oversampling (LR=0.001)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Leaf Blast | 0.97 | 0.98 | 0.98 |
| Brown Spot | 0.96 | 0.97 | 0.97 |
| Hispa | 0.98 | 0.97 | 0.98 |
| Healthy | 0.97 | 0.97 | 0.97 |
| **Overall** | **0.97** | **0.97** | **0.97** |

**Test Accuracy: 97.28% | Test Loss: 0.1563**

---

## Key Findings

- Triplet Attention improved test accuracy from **85.41% → 93.40%** (+7.99%)
- Random Oversampling further improved accuracy to **97.28%** (+3.88%)
- Brown Spot recall improved significantly: **0.61 → 0.97** after addressing class imbalance
- Training curves showed fluctuation, likely due to training from scratch without pretrained weights — identified as an area for future improvement

---

## Limitations & Future Work

- Model trained from scratch; using ImageNet pretrained weights may improve stability and convergence
- Only 4 of 10 available disease classes were used — extending to all classes is a natural next step
- Learning rate scheduling (e.g. ReduceLROnPlateau) could reduce training curve fluctuation
- Real-world deployment would require field-condition testing with varying lighting and image quality

---

## Repository Structure

```
├── ResNet50.ipynb           # Baseline model (ResNet-50 without attention)
└── ResNet50_TA.ipynb        # Proposed model (ResNet-50 + Triplet Attention + Oversampling)
```

---

## Framework & Environment

- **Framework**: TensorFlow / Keras
- **Platform**: Google Colab (T4 GPU)
- **Language**: Python 3

---

*Undergraduate thesis project — Informatics Engineering, Universitas Trunojoyo Madura (2025)*

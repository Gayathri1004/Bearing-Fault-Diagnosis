# Feature Extraction and Cost-Sensitive Learning for Bearing Fault Diagnosis with Imbalanced Data
This repository explores deep learning-based fault classification on two benchmark bearing datasets — CWRU and Paderborn — with a focus on tackling severe class imbalance. We investigate the role of feature extraction (spectrograms, scalograms) and cost-sensitive learning (class weights, focal loss) in improving generalization, especially for minority fault classes.

## Objective
- Study the effect of feature extraction techniques (time-frequency transforms) on fault classification performance
- Apply cost-sensitive learning techniques to handle extreme class imbalance
- Evaluate models across two datasets (CWRU and Paderborn) to assess generalizability

## Dataset
### CWRU Dataset
- Source: Case Western Reserve University
- Signals from accelerometers on drive-end and fan-end bearings
- Sample rates: 12 kHz and 48 kHz
- Fault Types: Inner race, Outer race, Ball (at various severity levels)
- Train-test splits designed with/without overlap and time-based separation
- Segments: e.g., 485,063 normal points vs 9,701 faulty — ~50:1 imbalance

### Paderborn Dataset
- Source: Paderborn University
- High-resolution vibration data (333.8 kHz sampling)
- Faults: Inner race, Outer race, rolling element (artificial + real faults)
- Each recording ~1.3M samples (4s)
- Segments generated via sliding window
- Extreme imbalance: normal class dominates fault classes by 50:1

## Preprocessing Pipeline
- Sliding window segmentation on raw 1D signals
- Feature extraction methods:
    - reshape to 2D
    - Spectrogram using STFT (scipy.signal.spectrogram)
    - Scalogram using CWT with Morlet wavelet (pywt.cwt)
- Normalization: Min-Max Scaling
- Resize all time-frequency maps to 96×96 or 40x40

## Model Architecture
- Convolutional Neural Network (CNN) with:
    - Stacked Conv2D layers + BatchNorm + MaxPooling
    - Dropout for regularization
    - Fully-connected Dense output layer
- Trained on:
  -  Raw (40×40 reshaped)
  -  Spectrograms (96×96)
  -  Scalograms (96×96)

## Cost-Sensitive Learning
- Class Weights computed based on inverse class frequencies
- Applied in:
  - Weighted Categorical Cross-Entropy
  - Focal Loss with α derived from class weights
  - Dynamic Focal Loss variant also tested on Paderborn

## Observations:
- Scalograms provided better fault separation due to adaptive resolution in both time and frequency
- Overlap improves performance but requires careful train-test separation (to avoid leakage)
- Cost-sensitive learning helped boost minority fault detection, especially on imbalanced test splits
- Focal loss didn't always outperform class weighting — tuning α/γ is crucial
- Time-based splitting is essential for realistic evaluation

## Project Status
✅ All experiments on CWRU and Paderborn completed
✅ Confusion matrices, F1 scores, and balanced accuracy evaluated
🔄 Exploring lightweight CNN variants for deployment scenarios

## Future Scope
- Ensemble model
- Explore transformer-based fault classifiers
- Use GANs or augmentation to boost minority data
- Real-time deployment feasibility

## Acknowledgements
This work was conducted as part of an MEng research project at the University of Waterloo, supervised by Dr. Kshirasagar Naik, Dr. Marzia Zaman, and Dr. Ravi Ravichandran. Datasets were sourced from Case Western Reserve University and Paderborn University.








# CodeAlpha-Task02
# Speech Emotion Recognition (SER) using CNNs

This project implements a Speech Emotion Recognition (SER) system using Python, Librosa, and TensorFlow. The model classifies emotions in speech audio based on features such as Mel Frequency Cepstral Coefficients (MFCCs).

## Overview

The system uses a Convolutional Neural Network (CNN) to classify audio speech samples into one of several emotional categories such as happy, sad, angry, or fearful. The input features are extracted from audio using MFCCs, and the model is trained on the RAVDESS dataset.

## Dataset

The project uses the [RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://zenodo.org/record/1188976) dataset.

- Each audio file is a short utterance spoken by actors expressing different emotions.
- Only the speech portion is used for emotion classification.

### Emotion Labels

| Label ID | Emotion    |
|----------|------------|
| 01       | Neutral    |
| 02       | Calm       |
| 03       | Happy      |
| 04       | Sad        |
| 05       | Angry      |
| 06       | Fearful    |
| 07       | Disgust    |
| 08       | Surprised  |

## Features

- MFCCs extracted using Librosa
- Each audio sample is padded or truncated to a fixed length
- Standardized inputs (zero-mean, unit variance)
- CNN architecture with Conv2D, MaxPooling, Dropout, and Dense layers

## Requirements

- Python 3.7+
- Google Colab (recommended)
- Libraries:
  - librosa
  - numpy
  - scikit-learn
  - matplotlib
  - tensorflow

## How to Run

1. Download and unzip the RAVDESS dataset
2. Upload it to Google Colab or your working directory
3. Extract MFCC features from each `.wav` file
4. Train the CNN model using the extracted features
5. Evaluate the model on test data

### Training the Model

```python
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Handwritten-Digit-Recognition-Using-SVD
Handwritten digit recognition using Singular Value Decomposition (SVD) for dimensionality reduction and classification on the MNIST dataset.

# Overview

This project demonstrates how **Singular Value Decomposition (SVD)** can be used to compress and represent image data while preserving the most significant features. The extracted features are then used to recognize handwritten digits.

The objective of this project is to illustrate the application of **linear algebra** in image processing and pattern recognition.

# Features

- Handwritten digit recognition
- Singular Value Decomposition (SVD)
- Image reconstruction
- Dimensionality reduction
- Feature extraction
- Python implementation

# Mathematical Background

# Singular Value Decomposition (SVD)

A matrix **A** can be decomposed as

```
A = UΣVᵀ
```

where

- **U** = Left singular vectors
- **Σ** = Diagonal matrix of singular values
- **Vᵀ** = Right singular vectors

By retaining only the largest singular values, the dimensionality of the data can be reduced while preserving its important features.

# Workflow

1. Load the handwritten digit image.
2. Convert the image into a matrix.
3. Apply Singular Value Decomposition.
4. Retain the dominant singular values.
5. Reconstruct the compressed image.
6. Recognize the handwritten digit.

# Results

# Test Image

![Test Digit](test_digit.png)

The project successfully demonstrates handwritten digit representation using SVD and highlights the effectiveness of dimensionality reduction for image-based pattern recognition.

# Technologies Used

- Python
- NumPy
- OpenCV
- Matplotlib

## Repository Structure

```text
Handwritten-Digit-Recognition-Using-SVD/
│
├── HandwrittenDigitRecognition.py
├── HandwrittenDigitRecognition_Report.pdf
├── test_digit.png
└── README.md
```

# Applications

- Optical Character Recognition (OCR)
- Image Compression
- Feature Extraction
- Pattern Recognition
- Machine Learning

# Future Improvements

- Train using the MNIST dataset
- Improve recognition accuracy
- Support multiple handwritten digits
- Build a graphical user interface
- Compare SVD with PCA and CNN-based approaches

# References

1. Gilbert Strang, *Linear Algebra and Its Applications*.
2. NumPy Documentation
3. OpenCV Documentation

# Author

**Revanth S**

B.Tech Artificial Intelligence & Data Science

Amrita Vishwa Vidyapeetham

⭐ If you found this project interesting, consider giving it a star.

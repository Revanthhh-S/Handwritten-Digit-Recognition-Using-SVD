import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import mnist
import cv2 # For reading custom images

# 1. Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Flatten images
X_train_flat = X_train.reshape(X_train.shape[0], 784).astype(np.float32) #0 represents the 1st dimension i.e 60,000 images
X_test_flat = X_test.reshape(X_test.shape[0], 784).astype(np.float32) #np.float32 for converting pixels from int to float

# 3. Normalize pixel values
X_train_flat /= 255.0 #We are dividing by 255.0 to make its range from 0.0-255.0 to 0.0-1.0
X_test_flat /= 255.0

# 4. Mean-center the data
mean_image = np.mean(X_train_flat, axis=0) #It calculates the average pixel value down each column.
X_train_centered = X_train_flat - mean_image
X_test_centered = X_test_flat - mean_image

# 5. Apply SVD
print("Performing SVD...")
U, s, Vt = np.linalg.svd(X_train_centered, full_matrices=False) #gives you the "economy" SVD

# 6. Select top k features
k = 50
V_k = Vt[:k, :]
# 7. Project training and test data
X_train_proj = X_train_centered @ V_k.T
X_test_proj = X_test_centered @ V_k.T

# 8. Train classifier
knn = KNeighborsClassifier(n_neighbors=1) #To find the single closest image from the training data and copy its label
knn.fit(X_train_proj, y_train)

# 10. Custom Image Input for Prediction
# ------------------------------------

def predict_custom_image(image_path):
    """
    Loads a grayscale image, preprocesses it, and predicts the digit.
    """
    # Load and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("❌ Error: Image not found or invalid path.")
        return

    # Resize to 28x28 if needed
    img_resized = cv2.resize(img, (28,28))
    # Invert colors if background is white
    if np.mean(img_resized) > 127:
        img_resized = 255 - img_resized

    # Normalize and flatten
    img_flat = img_resized.reshape(1, -1).astype(np.float32) / 255.0

    # Mean center using training mean
    img_centered = img_flat - mean_image

    # Project using SVD
    img_proj = img_centered @ V_k.T

    # Predict using KNN
    prediction = knn.predict(img_proj)[0]

    # Display the image and result
    plt.imshow(img_resized, cmap='gray')
    plt.title(f"Predicted digit: {prediction}")
    plt.axis('off')
    plt.show()

    return prediction
    
# Replace 'digit_sample.png' with the path to your digit image
your_image_path ="test_digit.png"
predict_custom_image("test_digit.png")

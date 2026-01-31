# ============================================================
# MNIST Digit Classification using a Neural Network
# Works in: VS Code / Local Jupyter Notebook
# ============================================================

# -------------------------------
# 1. Import required libraries
# -------------------------------

import numpy as np                     # Numerical computations
import matplotlib.pyplot as plt        # Data visualization
import seaborn as sns                  # Advanced visualizations
import cv2                             # Image processing
from PIL import Image                  # Image handling
import tensorflow as tf                # Deep learning framework

# Set random seed for reproducibility
tf.random.set_seed(3)

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.math import confusion_matrix


# -------------------------------
# 2. Load MNIST dataset
# -------------------------------
# MNIST contains handwritten digits (0–9)
# Training set: 60,000 images
# Test set: 10,000 images

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print("Training data shape:", X_train.shape, Y_train.shape)
print("Testing data shape :", X_test.shape, Y_test.shape)


# -------------------------------
# 3. Visualize sample image
# -------------------------------

plt.imshow(X_train[10], cmap="gray")
plt.title(f"Label: {Y_train[10]}")
plt.axis("off")
plt.show()


# -------------------------------
# 4. Data preprocessing
# -------------------------------
# Scale pixel values from [0,255] → [0,1]

X_train = X_train / 255.0
X_test  = X_test / 255.0


# -------------------------------
# 5. Build the Neural Network
# -------------------------------
# Architecture:
# Flatten → Dense(50) → Dense(50) → Dense(10)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),   # Convert 2D → 1D
    keras.layers.Dense(50, activation='relu'),    # Hidden layer
    keras.layers.Dense(50, activation='relu'),    # Hidden layer
    keras.layers.Dense(10, activation='sigmoid')  # Output layer
])


# -------------------------------
# 6. Compile the model
# -------------------------------

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# -------------------------------
# 7. Train the model
# -------------------------------

model.fit(X_train, Y_train, epochs=10)


# -------------------------------
# 8. Evaluate the model
# -------------------------------

loss, accuracy = model.evaluate(X_test, Y_test)
print("Test Accuracy:", accuracy)


# -------------------------------
# 9. Make predictions
# -------------------------------

Y_pred = model.predict(X_test)
Y_pred_labels = np.argmax(Y_pred, axis=1)


# -------------------------------
# 10. Confusion Matrix
# -------------------------------

conf_mat = confusion_matrix(Y_test, Y_pred_labels)

plt.figure(figsize=(12, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# -------------------------------
# 11. Predict a custom handwritten digit
# -------------------------------
# Image must be a handwritten digit on black background

image_path = input("Enter path of image: ")

# Read image
input_image = cv2.imread(image_path)

# Convert BGR → RGB for display
input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

plt.imshow(input_image_rgb)
plt.axis("off")
plt.show()

# Convert to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

# Resize to 28x28 (MNIST format)
resized_image = cv2.resize(gray_image, (28, 28))

# Normalize
resized_image = resized_image / 255.0

# Reshape for model input
image_reshaped = np.reshape(resized_image, (1, 28, 28))

# Prediction
prediction = model.predict(image_reshaped)
predicted_label = np.argmax(prediction)

print("The handwritten digit is recognized as:", predicted_label)

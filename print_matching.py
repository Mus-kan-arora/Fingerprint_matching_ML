import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns


# Load fingerprint dataset
x_data = np.load('x_real.npz')
print("x shape:", x_data.keys())  # Check the keys (image groups) in the loaded dataset

y_data = np.load('y_real.npy')  # Load labels
print("y shape:", y_data.shape)  # Shape of label data

# Extract the keys from x_data (group names in the dataset)
keys = list(x_data.keys())
print("First key:", keys[0])  # Print the first key


# Function to preprocess the fingerprint images (blurring, binarization, skeletonization)
def preprocess_fingerprint(image):
    # Apply Gaussian blur to smooth the image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Thresholding the image to binary (black & white)
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)
    binary_image = binary_image.astype(np.uint8)

    # Perform skeletonization to extract the ridges
    skeleton_image = cv2.ximgproc.thinning(binary_image)

    return skeleton_image


# Test the preprocessing function on the first image
example_image = x_data[keys[0]]
first_image = example_image[0, :, :, 0]  # Extract the first sample from the first group

# Preprocess the image
processed_image = preprocess_fingerprint(first_image)

# Print shapes of original and processed images
print("Original image shape:", example_image.shape)
print("Processed image shape:", processed_image.shape)

# Visualize the original and preprocessed images side-by-side
plt.subplot(1, 2, 1), plt.imshow(first_image, cmap='gray'), plt.title('Original Image'), plt.axis('off')
plt.subplot(1, 2, 2), plt.imshow(processed_image, cmap='gray'), plt.title('Preprocessed Image'), plt.axis('off')
plt.show()


# Function to extract minutiae points (ridge endings and bifurcations) from the skeletonized image
def extract_minutiae(skeleton_image):
    minutiae_points = []
    height, width = skeleton_image.shape

    # Loop through each pixel in the image
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if skeleton_image[x, y] == 255:  # If pixel is white (part of the ridge)
                neighborhood = skeleton_image[y - 1:y + 2, x - 1:x + 2]  # Get the 3x3 neighborhood
                count_neighbors = np.sum(neighborhood == 255)  # Count how many neighbors are white

                # If there are exactly 2 white neighbors, it's a ridge ending
                if count_neighbors == 2:
                    minutiae_points.append((x, y, 'ridge_ending'))

                # If there are exactly 4 white neighbors, it's a bifurcation point
                elif count_neighbors == 4:
                    minutiae_points.append((x, y, 'bifurcation'))

    return minutiae_points


# Function to extract the orientation field using Sobel operators
def extract_orientation_field(image):
    # Compute gradients in the x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # Calculate the angle of the gradient (orientation)
    angles = np.arctan2(sobel_y, sobel_x)

    # Convert angles from radians to degrees
    angles = np.degrees(angles)

    # Ensure all angles are positive
    angles[angles < 0] += 180

    return angles


# Function to normalize feature vectors
def normalize(features):
    min_val = np.min(features)
    max_val = np.max(features)

    # Handle case where min and max are the same (avoid division by zero)
    if max_val - min_val == 0:
        return features - min_val

    # Normalize the values to a range of [0, 1]
    normalized_features = (features - min_val) / (max_val - min_val)
    return normalized_features


# Convert minutiae points to a feature vector for classification
def minutiae_to_feature_vector(minutiae):
    feature_vector = []

    # Loop through each minutia point and encode it as [x, y, type]
    for minutia in minutiae:
        x, y, minutia_type = minutia
        if minutia_type == 'ridge_ending':
            feature_vector.append([x, y, 0])  # Ridge endings are labeled as 0
        elif minutia_type == 'bifurcation':
            feature_vector.append([x, y, 1])  # Bifurcations are labeled as 1

    return np.array(feature_vector)


# Extract features from the dataset
max_count = 40  # Maximum number of minutiae points to consider
feature_vectors = []

for img in x_data[keys[0]]:
    # Preprocess the image
    processed_image = preprocess_fingerprint(img)

    # Extract minutiae points
    minutiae = extract_minutiae(processed_image)

    # Convert minutiae points to a feature vector
    feature_vector = minutiae_to_feature_vector(minutiae)

    # Extract orientation field from the image
    orientation_field = extract_orientation_field(processed_image)

    # Handle cases with no minutiae points
    if feature_vector.size == 0:
        feature_vector = np.zeros((max_count, 3))

    # Pad or truncate the feature vector to a fixed length
    if feature_vector.shape[0] < max_count:
        padding = np.zeros((max_count - feature_vector.shape[0], 3))
        feature_vector = np.vstack((feature_vector, padding))
    else:
        feature_vector = feature_vector[:max_count]

    # Combine the minutiae feature vector with the orientation field
    combined_features = np.concatenate((feature_vector.flatten(), orientation_field.flatten()))

    # Normalize the combined feature vector
    normalized_features = normalize(combined_features)

    # Add the normalized feature vector to the list of feature vectors
    feature_vectors.append(normalized_features)


# Convert the list of feature vectors into a NumPy array
X = np.array(feature_vectors)

# Prepare labels for classification (flattened)
Y = np.array([label[1] for label in y_data]).flatten()


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Train a K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# Test the classifier on a single test image
test_image_vector = X_test[0]

# Predict the label using KNN
predicted_label = knn.predict([test_image_vector])
print("Predicted Label:", predicted_label)
print("Actual Label:", y_test[0])


# Evaluate the classifier on the entire test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Visualize the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Display the classification report (precision, recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred))

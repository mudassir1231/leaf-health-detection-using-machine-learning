import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import pickle


# Load the dataset
healthy_leaves = []
unhealthy_leaves = []


# Set the directory path where healthy leaf images are stored
healthy_leaves_dir = "./healthy leaf"
# make sure the images are of equal in number and have the same dimensions and pixels

# Iterate over the files in the directory
for filename in os.listdir(healthy_leaves_dir):
    # Create the file path
    file_path = os.path.join(healthy_leaves_dir, filename)

    # Load the image using OpenCV
    image = cv2.imread(file_path)

    # Append the loaded image to the healthy_leaves list
    healthy_leaves.append(image)

unhealthy_leaves_dir = "./unhealthy leaf"

# Iterate over the files in the directory
for filename in os.listdir(unhealthy_leaves_dir):
    # Create the file path
    file_path = os.path.join(unhealthy_leaves_dir, filename)

    # Load the image using OpenCV
    image = cv2.imread(file_path)

    # Append the loaded image to the healthy_leaves list
    unhealthy_leaves.append(image)


# Load healthy leaf images
# Append the loaded images to the healthy_leaves list

# Load unhealthy leaf images
# Append the loaded images to the unhealthy_leaves list

# Create labels for the dataset
healthy_labels = np.zeros(len(healthy_leaves))
unhealthy_labels = np.ones(len(unhealthy_leaves))

print(len(unhealthy_leaves))
print(len(healthy_leaves))
print(unhealthy_labels)

# Combine the data and labels
X = np.concatenate((healthy_leaves, unhealthy_leaves))
y = np.concatenate((healthy_labels, unhealthy_labels))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Flatten the images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=900, random_state=50)
clf.fit(X_train_flat, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_flat)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Load a test image
test_image = cv2.imread("./0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417_new30degFlipLR.JPG")

# Preprocess the test image (resize, convert to grayscale, etc.)
# ...

# Extract features from the preprocessed image (flatten or use feature extraction methods)
# ...

# Flatten the test image
test_image_flat = test_image.reshape(1, -1)

prediction = clf.predict(test_image_flat)

# Make a prediction on the test image

# Assuming 'model' is your trained machine learning model
model = clf  # Your trained model

# Save the model to a file
filename = 'model2.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
print("saved")



# Display the prediction
if prediction == 0:
    print("The leaf is healthy.")
else:
    print("The leaf is unhealthy.")

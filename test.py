import numpy as np
import pickle
import cv2
import sklearn
# Load the model from the file
filename = 'model2.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)
print("done")

# test_image = cv2.imread("./0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417_new30degFlipLR.JPG")

# # Preprocess the test image (resize, convert to grayscale, etc.)
# # ...

# # Extract features from the preprocessed image (flatten or use feature extraction methods)
# # ...

# # Flatten the test image
# test_image_flat = test_image.reshape(1, -1)

# prediction = model.predict(test_image_flat)
# # Make a prediction on the test image


# # Display the prediction


# Set up the camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera

# Desired input size of your model
input_size = (256, 256)

# Loop for capturing and processing frames
while True:
    # Read frame from the camera
    ret, frame = camera.read()

    # Resize the frame to match the input size of your model
    resized_frame = cv2.resize(frame, input_size)

    # Preprocess the resized frame if needed (e.g., normalize, etc.)
    # preprocessed_frame = resized_frame.reshape(1, -1)  # Your preprocessing steps
    flattened_frame = resized_frame.reshape(1, -1)

    # Pass the preprocessed frame through your trained model
    prediction = model.predict(flattened_frame)

    probabilities = model.predict_proba(flattened_frame)
    healthy_prob = probabilities[0, 0]
    unhealthy_prob = probabilities[0, 1]

    print(healthy_prob,unhealthy_prob)
    # Print the prediction and confidence scores
    if healthy_prob * 100 > 90:
        print("LEAF healthy with a confidence of {:.2f}%".format(healthy_prob * 100))
    elif unhealthy_prob * 100 > 90:
        print("LEAF unhealthy with a confidence of {:.2f}%".format(unhealthy_prob * 100))
    # else:
        # print("no leaf found")
    # Process the prediction (e.g., determine plant health status)

    # Visualize the output on the frame or perform any desired actions

    # Display the frame with the output
    cv2.imshow('Live Camera', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()

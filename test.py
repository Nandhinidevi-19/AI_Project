import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Define the path to the saved model
model_path = r'C:\Users\curse\OneDrive\Desktop\AI_Project\cnn_model.keras'

# Load the saved model
model = load_model(model_path)

# Function to preprocess a single test image
def preprocess_single_image(image_path, target_size=(192, 192)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img = img.convert('RGB')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Define the path to the single test image
test_image_path = r'C:\Users\curse\OneDrive\Desktop\AI_Project\Original\2.png'  # Replace with your test image path

# Ensure the test image path is valid
if not os.path.isfile(test_image_path):
    raise ValueError(f"The file {test_image_path} does not exist or is invalid.")

# Preprocess the single test image
test_image = preprocess_single_image(test_image_path)

# Check if preprocessing was successful
if test_image is None:
    raise ValueError("Preprocessing failed. Please check the input image.")

# Debug: Print the shape of the test image array
print(f"Shape of test image array: {test_image.shape}")

# Rescale the image
test_image_rescaled = test_image / 255.0

# Predict on the single test image
y_pred = model.predict(test_image_rescaled)
y_pred_class = (y_pred > 0.5).astype("int32")

# Print the prediction result
if y_pred_class[0][0] == 0:
    print("The image is original.")
else:
    print("The image is recolored.")
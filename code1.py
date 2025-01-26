import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define the path to the UCID dataset
dataset_path = r'C:\Users\curse\OneDrive\Desktop\AI_Project\UCID1338'
original_folder = r'C:\Users\curse\OneDrive\Desktop\AI_Project\Original'
recolored_folder = r'C:\Users\curse\OneDrive\Desktop\AI_Project\recolored'

# Create folders if they don't exist
os.makedirs(original_folder, exist_ok=True)
os.makedirs(recolored_folder, exist_ok=True)

# Function to resize and convert images to PNG
def preprocess_images(image_paths, target_size=(192, 192)):
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            img = img.resize(target_size)
            img = img.convert('RGB')
            new_path = os.path.join(original_folder, os.path.basename(img_path).replace('.tif', '.png'))
            img.save(new_path, 'PNG')  # Save as PNG
            images.append(np.array(img))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return np.array(images)

# Load image paths
image_paths = [os.path.join(dataset_path, fname) for fname in os.listdir(dataset_path) if fname.lower().endswith('.tif')]

# Select only 2676 images
selected_image_paths = image_paths[:2676]

# Debug: Print the number of images found
print(f"Number of images selected: {len(selected_image_paths)}")

# Debug: Print the first few image paths to verify
if selected_image_paths:
    print(f"First few image paths: {selected_image_paths[:5]}")

# Preprocess images
images = preprocess_images(selected_image_paths)

# Debug: Print the shape of the images array
print(f"Shape of images array: {images.shape}")

# Recolorize images using a color inversion filter (not grayscale)
recolored_images = np.array([cv2.bitwise_not(img) for img in images])  # Invert colors

# Save recolored images
for i, img in enumerate(recolored_images):
    recolored_path = os.path.join(recolored_folder, f'recolored_{i}.png')
    Image.fromarray(img).save(recolored_path, 'PNG')

# Debug: Print the shape of the recolored_images array
print(f"Shape of recolored_images array: {recolored_images.shape}")

# Ensure both arrays have the same shape
if images.shape[1:] != recolored_images.shape[1:]:
    raise ValueError(f"Shape mismatch: images shape {images.shape}, recolored_images shape {recolored_images.shape}")

# Combine original and recolored images
all_images = np.concatenate((images, recolored_images), axis=0)
labels = np.concatenate((np.zeros(len(images)), np.ones(len(recolored_images))), axis=0)

# Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(all_images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
train_generator = datagen.flow(X_train, y_train, subset='training', batch_size=16)  # Smaller batch size
val_generator = datagen.flow(X_val, y_val, subset='validation', batch_size=16)  # Smaller batch size

## Step 2: Build the CNN Model

# Define the CNN architecture
model = Sequential([
    Input(shape=(192, 192, 3)),  # Adjusted input shape
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the model using the native Keras format
model.save('cnn_model.keras')

## Step 3: Evaluate the Model

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
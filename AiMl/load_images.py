import cv2
from glob import glob
import numpy as np
from sklearn.utils import shuffle
import pickle
import os

def pickle_images_labels():
    images_labels = []
    # Correct glob pattern to recursively find all .jpg files
    images = glob("gestures/**/*.jpg", recursive=True)
    images.sort()
    for image in images:
        print(image)
        # Extract label based on the directory name
        label = image.split(os.sep)[-2]  # Assumes label is the second last directory name
        img = cv2.imread(image, 0)
        images_labels.append((np.array(img, dtype=np.uint8), int(label)))
    return images_labels

# Load images and labels
images_labels = pickle_images_labels()

# Shuffle the dataset
images_labels = shuffle(images_labels)

# Unpack into images and labels
images, labels = zip(*images_labels)
total_samples = len(images_labels)
print(f"Total samples: {total_samples}")

# Define split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate indices for splitting
train_idx = int(train_ratio * total_samples)
val_idx = int((train_ratio + val_ratio) * total_samples)

# Split the dataset into training, validation, and testing sets
train_images, train_labels = images[:train_idx], labels[:train_idx]
val_images, val_labels = images[train_idx:val_idx], labels[train_idx:val_idx]
test_images, test_labels = images[val_idx:], labels[val_idx:]

# Save the datasets as pickle files
with open("train_images.pkl", "wb") as f:
    pickle.dump(train_images, f)
with open("train_labels.pkl", "wb") as f:
    pickle.dump(train_labels, f)

with open("val_images.pkl", "wb") as f:
    pickle.dump(val_images, f)
with open("val_labels.pkl", "wb") as f:
    pickle.dump(val_labels, f)

with open("test_images.pkl", "wb") as f:
    pickle.dump(test_images, f)
with open("test_labels.pkl", "wb") as f:
    pickle.dump(test_labels, f)

print(f"Training set: {len(train_images)} images")
print(f"Validation set: {len(val_images)} images")
print(f"Test set: {len(test_images)} images")


# Define file paths
file_paths = [
    "train_images.pkl",
    "train_labels.pkl",
    "val_images.pkl",
    "val_labels.pkl",
    "test_images.pkl",
    "test_labels.pkl"
]

# Print the sizes of each pickle file
for file_path in file_paths:
    size = os.path.getsize(file_path)
    print(f"Size of {file_path}: {size} bytes")

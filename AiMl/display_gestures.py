import cv2
import os
import numpy as np
import random

def get_image_size():
    """Get the size of a sample image."""
    img = cv2.imread('gestures/1/100.jpg', 0)
    return img.shape

def get_folders():
    """Retrieve all folder names from the 'gestures' directory."""
    return [folder for folder in os.listdir('gestures') if os.path.isdir(os.path.join('gestures', folder))]

def create_image_grid(images, image_size, num_images_per_row):
    """Create a grid of images."""
    image_x, image_y = image_size
    num_images = len(images)
    rows = (num_images + num_images_per_row - 1) // num_images_per_row  # Calculate number of rows
    full_img = np.zeros((rows * image_y, num_images_per_row * image_x), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // num_images_per_row
        col = idx % num_images_per_row
        full_img[row * image_y:(row + 1) * image_y, col * image_x:(col + 1) * image_x] = img

    return full_img

def process_images_from_folder(folder, image_size):
    """Process images from a given folder and return a list of images."""
    images = []
    for i in range(1200):  # Assuming 1200 images; adjust as needed
        img_path = f"gestures/{folder}/{i+1}.jpg"
        img = cv2.imread(img_path, 0)
        if img is None:
            img = np.zeros(image_size, dtype=np.uint8)  # Create a blank image if the file is missing
        images.append(img)
    return images

def main():
    folders = get_folders()
    
    if not folders:
        print("No folders found in the 'gestures' directory.")
        return
    
    if len(folders) == 1:
        folder = folders[0]
        print(f"Processing single folder: {folder}")
        images = process_images_from_folder(folder, get_image_size())
    else:
        print(f"Processing multiple folders: {folders}")
        images = []
        for folder in folders:
            images.extend(process_images_from_folder(folder, get_image_size()))
    
    # Shuffle images
    random.shuffle(images)
    
    # Get image size and create grid
    image_size = get_image_size()
    num_images_per_row = 5  # Number of images per row
    full_img = create_image_grid(images, image_size, num_images_per_row)
    
    # Display and save the concatenated image
    cv2.imshow("gestures", full_img)
    cv2.imwrite('full_img.jpg', full_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

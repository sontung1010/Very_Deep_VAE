import cv2
import os
import numpy as np

## This is to applying our code to custom dataset
# =======================================================================

def crop_and_resize_images(input_folder, output_folder, output_size=(32, 32)):
    ## This is for resizing the input data (Resizing for faster training)
    # ==========================================================================
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = [f for f in os.listdir(input_folder) if f.endswith('.JPEG')] #.jpg)]

    for image_name in images:
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to load {image_name}, skipping...")
            continue
        h, w, _ = image.shape
        if h > w:
            crop_size = w
            top = (h - w) // 2
            left = 0
            cropped = image[top:top + crop_size, left:left + crop_size]
        else:
            crop_size = h
            top = 0
            left = (w - h) // 2
            cropped = image[top:top + crop_size, left:left + crop_size]

        # Resize to 64x64
        resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)
        output_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + '.png')
        cv2.imwrite(output_path, resized)
        print(f"saved: {output_path}")

input_folder = 'imagenet_filtered/original'
output_folder = 'imagenet_filtered/processed'
crop_and_resize_images(input_folder, output_folder)

def calculate_shift_and_scale_with_cv2(folder_path):
    total_sum = np.zeros(3) 
    total_squared_sum = np.zeros(3) 
    total_pixels = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path) 
            if image is None:
                print(f"Failed {file_name}, skipping")
                continue
            image = image.astype(np.float32)

            total_sum += image.sum(axis=(0, 1)) 
            total_squared_sum += (image ** 2).sum(axis=(0, 1))  
            total_pixels += image.shape[0] * image.shape[1] 

    mean = total_sum / total_pixels
    std = np.sqrt((total_squared_sum / total_pixels) - mean ** 2)

    return np.mean(mean), np.mean(std)

print(calculate_shift_and_scale_with_cv2(output_folder))
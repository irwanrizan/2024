from PIL import Image
import numpy as np
import os
import cv2

def remove_blue_color(image):

    # Convert BGR to HSV
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
    low_blue = np.array([90, 50, 50])    # Lower HSV values for blue
    high_blue = np.array([130, 255, 255])  # Higher HSV values for blue
    
    # Create mask for blue color
    mask = cv2.inRange(imgHSV, low_blue, high_blue)
    
    # Perform morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Replace blue pixels with average of neighboring pixels
    blurred = cv2.GaussianBlur(image, (15, 15), 0)  # Blur image to reduce noise
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j] == 255:  # If pixel is blue
                neighbor_color = np.mean(blurred[max(0, i-10):min(image.shape[0], i+10), max(0, j-10):min(image.shape[1], j+10)], axis=(0, 1)).astype(np.uint8)
                image[i, j] = neighbor_color
    
    return image
    

def crop_image_based_on_position(input_image_path, output_image_path, top_frame_height, bottom_frame_height, right_frame_width, left_frame_width):
    try:
        # Read the image
        img = cv2.imread(input_image_path)
        if img is None:
            raise ValueError(f"Image not found or unable to read: {input_image_path}")
        
        # Get the dimensions of the image
        height, width, _ = img.shape
        # Define the crop area
        crop_img = img[top_frame_height:height-bottom_frame_height, left_frame_width:width-right_frame_width]
        # Remove blue color from the cropped image
        ##no_blue_img = remove_blue_color(crop_img) # Skip this if dont want to remove any blue color
        # Save the processed image
        cv2.imwrite(output_image_path, crop_img)
        print(f'Cropped image with blue removed saved to {output_image_path}')
    except Exception as e:
        print(f"An error occurred: {e}")

Direction = 'Right'


input_image = r"C:\Users\Irwan\Desktop\New folder\Dataset\SAW IMAGE\Pass"
output_folder = r"C:\Users\Irwan\Desktop\New folder\Dataset\SAW IMAGE\Pass\Crop-Images-"+Direction

if Direction == 'Both':
    top_frame_height = 700    
    bottom_frame_height = 200  
    right_frame_width = 250
    left_frame_width = 10
elif Direction == 'Left':
    top_frame_height = 705
    bottom_frame_height = 190
    right_frame_width = 770
    left_frame_width = 10
elif Direction == 'Right':
    top_frame_height = 703
    bottom_frame_height = 192
    right_frame_width = 250
    left_frame_width = 530

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_image):
    name, ext = os.path.splitext(filename)
    input_image_path = os.path.join(input_image, filename)
    output_image_path = os.path.join(output_folder, name+'-'+Direction+ext)
    crop_image_based_on_position(input_image_path, output_image_path, top_frame_height, bottom_frame_height, right_frame_width, left_frame_width)



   


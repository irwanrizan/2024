import numpy as np
import os
import cv2

def remove_blue_color(image):
    """
    Remove blue color from an image.
    
    Args:
        image (numpy.ndarray): The input image in BGR format.
    
    Returns:
        numpy.ndarray: The image with blue color removed.
    """
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
    low_blue = np.array([90, 50, 50])    # Lower HSV values for blue
    high_blue = np.array([130, 255, 255])  # Higher HSV values for blue
    
    mask = cv2.inRange(imgHSV, low_blue, high_blue)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    blurred = cv2.GaussianBlur(image, (15, 15), 0)  # Blur image to reduce noise
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if mask[i, j] == 255:  # If pixel is blue
                neighbor_color = np.mean(blurred[max(0, i-10):min(image.shape[0], i+10), max(0, j-10):min(image.shape[1], j+10)], axis=(0, 1)).astype(np.uint8)
                image[i, j] = neighbor_color
    
    return image

def crop_image_based_on_position(input_image_path, output_image_path, top_frame_height, bottom_frame_height, right_frame_width, left_frame_width, remove_blue=False):
    """
    Crop an image based on given frame positions and optionally remove blue color.
    
    Args:
        input_image_path (str): Path to the input image.
        output_image_path (str): Path to save the cropped image.
        top_frame_height (int): Pixels to crop from the top.
        bottom_frame_height (int): Pixels to crop from the bottom.
        right_frame_width (int): Pixels to crop from the right.
        left_frame_width (int): Pixels to crop from the left.
        remove_blue (bool): If True, remove blue color from the cropped image.
    """
    try:
        img = cv2.imread(input_image_path)
        if img is None:
            raise ValueError(f"Image not found or unable to read: {input_image_path}")
        
        height, width, _ = img.shape
        crop_img = img[top_frame_height:height-bottom_frame_height, left_frame_width:width-right_frame_width]
        
        if remove_blue:
            crop_img = remove_blue_color(crop_img)
        
        cv2.imwrite(output_image_path, crop_img)
        print(f'Cropped image saved to {output_image_path}')
    except Exception as e:
        print(f"An error occurred: {e}")

def main(input_image_folder, output_folder, direction, remove_blue=False):
    """
    Main function to process images in a folder based on direction and crop parameters.
    
    Args:
        input_image_folder (str): Path to the input image folder.
        output_folder (str): Path to save the processed images.
        direction (str): Direction for cropping. Options: 'Both', 'Left', 'Right'.
        remove_blue (bool): If True, remove blue color from the cropped images.
    """
    directions = {
        'Both': (700, 200, 250, 10),
        'Left': (705, 190, 770, 10),
        'Right': (703, 192, 250, 530)
    }
    
    if direction not in directions:
        raise ValueError(f"Invalid direction: {direction}. Valid options are 'Both', 'Left', 'Right'")
    
    top_frame_height, bottom_frame_height, right_frame_width, left_frame_width = directions[direction]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_image_folder):
        name, ext = os.path.splitext(filename)
        input_image_path = os.path.join(input_image_folder, filename)
        output_image_path = os.path.join(output_folder, name + '-' + direction + ext)
        crop_image_based_on_position(input_image_path, output_image_path, top_frame_height, bottom_frame_height, right_frame_width, left_frame_width, remove_blue)

if __name__ == "__main__":
    input_image_folder = r"C:\Users\Irwan\Desktop\New folder\Dataset\SAW IMAGE\Pass"
    output_folder = r"C:\Users\Irwan\Desktop\New folder\Dataset\SAW IMAGE\Pass\Crop-Images-Right"
    direction = 'Right'
    
    main(input_image_folder, output_folder, direction, remove_blue=False)

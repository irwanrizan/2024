import cv2
import os
import numpy as np

# Global variables
threshold1_value = 40#36#54  # Initial threshold
threshold2_value = 60#80#60  # Initial threshold

def draw_dashed_line(image, y, color=(0, 255, 0), thickness=1, dash_length=5):

    x1, x2 = 0, image.shape[1]
    is_dash = True
    for x in range(x1, x2, dash_length * 2):
        if is_dash:
            cv2.line(image, (x, y), (x + dash_length, y), color, thickness)
        is_dash = not is_dash

def apply_dash_line(image, contours):
    if image is None:
        print('Error: No image')
        return

    lowest_y = image.shape[0]
    highest_y = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if y >= 35:  
            if y < lowest_y:
                lowest_y = y
        if y + h <= image.shape[0] - 35:  
            if y + h > highest_y:
                highest_y = y + h

    cv2.rectangle(image, (0, lowest_y), (image.shape[1]-1, highest_y), (0, 255, 0), 1)

    middle_y = (lowest_y + highest_y) // 2
    draw_dashed_line(image, middle_y)

    cv2.imwrite(output_path, image)

    return image

def apply_threshold(input_path):
    global threshold1_value, threshold2_value

    image = cv2.imread(input_path)
    if image is None:
        print(f'Error: Unable to read image from {input_path}')
        return
    
    contrast = 1
    bright = 5
    new_image = cv2.convertScaleAbs(image, alpha=contrast, beta=bright)
    
    lab = cv2.cvtColor(new_image, cv2.COLOR_BGR2Lab)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    new_image = np.hstack((new_image, enhanced_img))

    gray_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.dilate(gray_image, kernel=np.ones((4,4),np.uint8), iterations=1)
    gray_image = cv2.erode(gray_image, kernel=np.ones((2,2),np.uint8), iterations=1)

    gray_image = cv2.bilateralFilter(gray_image,15,85,85)

    #kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(gray_image, -1, kernel)

    _, binary_image1 = cv2.threshold(sharpened_image, threshold1_value, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(sharpened_image, threshold2_value, 255, cv2.THRESH_BINARY_INV)

    combined_image = cv2.bitwise_and(binary_image1, binary_image2)
    contours, _ = cv2.findContours(combined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Apply bounding box
    result_image = apply_dash_line(image.copy(), contours)

    path, name = os.path.split(input_path)

    cv2.imshow('Result', result_image)
    cv2.waitKey(100)

input_image = r"C:\Users\Irwan\Desktop\New folder\Dataset\SAW IMAGE\Not Pass\Not Pass-Crop"
output_folder = r"C:\Users\Irwan\Desktop\New folder\Dataset\SAW IMAGE\Not Pass\Crop-Images-Right\BBOX"

if not os.path.exists(input_image):
    os.makedirs(input_image)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def is_image_file(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return os.path.splitext(filename)[1].lower() in image_extensions

for filename in os.listdir(input_image):
    if is_image_file(filename):
        input_path = os.path.join(input_image, filename)
        output_path = os.path.join(output_folder, filename)
        apply_threshold(input_path)

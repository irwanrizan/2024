import cv2
import numpy as np

# Global variables
threshold1_value = 36  # Initial threshold
threshold2_value = 80  # Initial threshold
image_path = r'C:\Users\Irwan\Desktop\New folder\Dataset\SAW IMAGE\Not Pass\Not Pass-Crop\F28-Right.png'

# Flag to toggle between views
view_mode = 0  # 0 - Thresholded, 1 - Segmented, 2 - Bounding Box
segmented_image = None  # Global variable to store segmented image

# Function to perform segmentation based on thresholds
def perform_segmentation(image, threshold1, threshold2):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, binary_image1 = cv2.threshold(gray_image, threshold1, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(gray_image, threshold2, 255, cv2.THRESH_BINARY_INV)

    # Combine the two thresholded images for visualization
    combined_image = cv2.bitwise_and(binary_image1, binary_image2)

    # Find contours in the combined image
    contours, _ = cv2.findContours(combined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw the segmented regions
    segmented = np.zeros_like(image)

    # Draw each contour (segmented region) with a random color
    for i, contour in enumerate(contours):
        color = tuple(np.random.randint(0, 256, 3).tolist())  # Random color for each contour
        cv2.drawContours(segmented, contours, i, color, -1)  # Fill contour with color
    
    return segmented

def on_trackbar1(value):
    global threshold1_value
    threshold1_value = value
    if view_mode == 1:
        apply_segmentation()
    elif view_mode == 2:
        apply_bounding_boxes()
    else:
        apply_threshold()  # Update thresholded image

def on_trackbar2(value):
    global threshold2_value
    threshold2_value = value
    if view_mode == 1:
        apply_segmentation()
    elif view_mode == 2:
        apply_bounding_boxes()
    else:
        apply_threshold()  # Update thresholded image

def apply_threshold():
    global image_path, threshold1_value, threshold2_value
    
    # Read the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Could not load image.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened_image = cv2.filter2D(gray_image, -1, kernel)

    _, binary_image1 = cv2.threshold(sharpened_image, threshold1_value, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(sharpened_image, threshold2_value, 255, cv2.THRESH_BINARY_INV)

    combined_image = cv2.bitwise_and(binary_image1, binary_image2)
    contours, _ = cv2.findContours(combined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Display the thresholded image
    cv2.imshow('Result', combined_image)

def apply_segmentation():
    global image_path, threshold1_value, threshold2_value, segmented_image
    
    # Read the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Could not load image.")
        return

    # Perform segmentation based on current threshold values
    segmented_image = perform_segmentation(original_image, threshold1_value, threshold2_value)

    # Display the segmented image
    cv2.imshow('Result', segmented_image)

def apply_bounding_boxes():
    global image_path, threshold1_value, threshold2_value
    
    # Read the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Could not load image.")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, binary_image1 = cv2.threshold(gray_image, threshold1_value, 255, cv2.THRESH_BINARY)
    _, binary_image2 = cv2.threshold(gray_image, threshold2_value, 255, cv2.THRESH_BINARY_INV)

    # Combine the two thresholded images for visualization
    combined_image = cv2.bitwise_and(binary_image1, binary_image2)

    # Find contours in the combined image
    contours, _ = cv2.findContours(combined_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow('Result', original_image)

# Load your image initially
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image.")
else:
    # Create a window to display the results
    cv2.namedWindow('Result')

    # Create trackbars (scroll bars) for both thresholds
    cv2.createTrackbar('Threshold 1', 'Result', threshold1_value, 255, on_trackbar1)
    cv2.createTrackbar('Threshold 2', 'Result', threshold2_value, 255, on_trackbar2)

    # Apply initial view (thresholded image)
    apply_threshold()

    while True:
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # Toggle between views using 't' key
        if key == ord('t'):
            view_mode = (view_mode + 1) % 3  # Cycle through 0, 1, 2
            if view_mode == 0:
                apply_threshold()
            elif view_mode == 1:
                if segmented_image is None:
                    apply_segmentation()
                else:
                    cv2.imshow('Result', segmented_image)
            elif view_mode == 2:
                apply_bounding_boxes()

        # Exit loop if 'q' key is pressed
        elif key == ord('q'):
            break

    # Close all windows
    cv2.destroyAllWindows()
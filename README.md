# Image Processing Toolkit

This repository contains a set of tools for image processing, particularly focusing on cropping images based on frame positions and optionally removing blue color from the images. The project is implemented using Python and OpenCV.

## Features

- **Remove Blue Color**: A function to remove blue color from images by replacing blue pixels with the average color of neighboring pixels.
- **Crop Images**: A function to crop images based on specified frame positions (top, bottom, right, left).
- **Configurable Parameters**: Easily configurable parameters for cropping dimensions and whether to remove blue color.
- **Batch Processing**: Process all images in a specified input folder and save the processed images to an output folder.
- **Error Handling**: Robust error handling to manage invalid inputs and missing files.

## Installation

Ensure you have the required libraries installed:

```bash
pip install numpy opencv-python
```

# Usage
## 1. Configuration:
Modify the main function parameters as needed:
```
input_image_folder = r"path\to\input\folder"
output_folder = r"path\to\output\folder"
direction = 'Right'  # Options: 'Both', 'Left', 'Right'
remove_blue = False  # Set to True if you want to remove blue color
```
## 2. Run the Script:
Execute the script to process the images:
```
python Crop_image.py
```

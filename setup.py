import cv2
import matplotlib.pyplot as plt
import numpy as np
from webcolors import rgb_to_name  # Make sure to install webcolors: pip install webcolors

# Load the image
img = cv2.imread('examples/test1.jpg')  # Replace with your image file path

# Convert the image from BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape the image to a list of pixels
pixels = img_rgb.reshape((-1, 3))

# Calculate the histogram
hist = np.histogramdd(pixels, bins=(256, 256, 256), range=((0, 256), (0, 256), (0, 256)))[0]

# Normalize the histogram
hist /= hist.sum()

# Find the dominant color (the bin with the maximum probability)
dominant_color = np.unravel_index(hist.argmax(), hist.shape)

# Convert the dominant color from 8-bit to 0-1 scale
dominant_color_normalized = [c / 255.0 for c in dominant_color]

# Display the dominant color
print(f'Dominant Color (RGB): {dominant_color}')

try:
    dominant_color_name = rgb_to_name(dominant_color)
except ValueError:
    # Handle the case where the color doesn't have a predefined name
    dominant_color_name = f'Unknown Color (RGB: {dominant_color})'

print(f'Dominant Color (Name): {dominant_color_name}')

# Define a color tolerance for matching colors (10%)
color_tolerance = 0.1  # Adjust as needed

# Define the color range for the bounding box
lower_bound = np.array([c - color_tolerance * 255 for c in dominant_color])
upper_bound = np.array([c + color_tolerance * 255 for c in dominant_color])

# Threshold the image to get a binary mask
mask = cv2.inRange(img_rgb, lower_bound, upper_bound)

# Find contours in the binary mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to store information about the initial bounding box
initial_area = 0
initial_x, initial_y, initial_w, initial_h = 0, 0, 0, 0

# Iterate through contours to find the initial bounding box
for contour in contours:
    area = cv2.contourArea(contour)
    if area > initial_area:
        initial_area = area
        initial_x, initial_y, initial_w, initial_h = cv2.boundingRect(contour)

# User input for image dimensions
actual_width_meters = float(input("Enter actual width of the image in meters: "))
actual_height_meters = float(input("Enter actual height of the image in meters: "))
actual_area_meters = float(input("Enter actual area of the image in square meters: "))

# Calculate the conversion factor from pixels to meters using the square root of the area
pixels_per_meter_area = np.sqrt(initial_area) / np.sqrt(actual_area_meters)

# Convert pixel dimensions to meters
initial_width_meters = initial_w / pixels_per_meter_area
initial_height_meters = initial_h / pixels_per_meter_area

# Calculate initial deviations
initial_width_deviation = abs(initial_width_meters - actual_width_meters) / actual_width_meters * 100
initial_height_deviation = abs(initial_height_meters - actual_height_meters) / actual_height_meters * 100

# Display the initial bounding box with text using matplotlib
cv2.rectangle(img_rgb, (initial_x, initial_y), (initial_x + initial_w, initial_y + initial_h), (0, 255, 0), 2)
initial_text = f'Initial Height: {initial_h}, Width: {initial_w}, Area: {initial_area}, Image Area: {actual_area_meters}, Dominant Color: {dominant_color_name}'
initial_text_position = (max(0, initial_x - 10), max(30, initial_y - 10))
cv2.putText(img_rgb, initial_text, initial_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
plt.imshow(img_rgb)
plt.title('Initial Bounding Box with Dimensions and Dominant Color')
plt.axis('off')
plt.show()

# Calculate adjustments based on initial deviations
adjustment_factor = 0.1  # Adjust as needed
adjusted_width = int(initial_w * (1 + initial_width_deviation / 100 * adjustment_factor))
adjusted_height = int(initial_h * (1 + initial_height_deviation / 100 * adjustment_factor))

# Draw the adjusted bounding box on the image
cv2.rectangle(img_rgb, (initial_x, initial_y), (initial_x + adjusted_width, initial_y + adjusted_height), (0, 255, 0), 2)

# Add text to display adjusted height, width, area of the bounding box, area of the image, and dominant color
adjusted_text = f'Adjusted Height: {adjusted_height}, Width: {adjusted_width}, Area: {initial_area}, Image Area: {actual_area_meters}, Dominant Color: {dominant_color_name}'
adjusted_text_position = (max(0, initial_x - 10), max(30, initial_y - 10))

# Print the adjusted values
print(f'Adjusted Height: {adjusted_height}, Width: {adjusted_width}, Area: {initial_area}, Image Area: {actual_area_meters}, Dominant Color: {dominant_color_name}')

# Display the adjusted bounding box with text using matplotlib
cv2.putText(img_rgb, adjusted_text, adjusted_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
plt.imshow(img_rgb)
plt.title('Adjusted Bounding Box with Dimensions and Dominant Color')
plt.axis('off')
plt.show()

import cv2
import matplotlib.pyplot as plt
import numpy as np
from webcolors import rgb_to_name  # Make sure to install webcolors: pip install webcolors

# Load the image
img = cv2.imread('examples/test2.jpg') 
# Get image dimensions
img_height, img_width, _ = img.shape

# Set the initial bounding box factor
initial_bbox_factor = 0.8

# Calculate the dimensions for the initial bounding box
bounding_box_width = int(img_width * initial_bbox_factor)
bounding_box_height = int(img_height * initial_bbox_factor)

# Calculate the coordinates for the top-left corner of the initial bounding box
x = (img_width - bounding_box_width) // 2
y = (img_height - bounding_box_height) // 2

# Create an empty mask for the initial bounding box
initial_mask = np.zeros((img_height, img_width), dtype=np.uint8)
cv2.rectangle(initial_mask, (x, y), (x + bounding_box_width, y + bounding_box_height), 255, thickness=cv2.FILLED)

# Apply the initial mask to the original image
initial_result = cv2.bitwise_and(img, img, mask=initial_mask)

# Convert the initial result to grayscale
initial_gray = cv2.cvtColor(initial_result, cv2.COLOR_BGR2GRAY)

# Perform edge detection using Canny on the initial result
initial_edged = cv2.Canny(initial_gray, 50, 100)
initial_edged = cv2.dilate(initial_edged, None, iterations=1)
initial_edged = cv2.erode(initial_edged, None, iterations=1)

# Find contours in the initial result
initial_contours, _ = cv2.findContours(initial_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to store information about the largest bounding box in the initial result
initial_largest_area = 0
initial_largest_x, initial_largest_y, initial_largest_w, initial_largest_h = 0, 0, 0, 0

# Iterate through contours to find the largest bounding box in the initial result
for contour in initial_contours:
    area = cv2.contourArea(contour)
    if area > initial_largest_area:
        initial_largest_area = area
        initial_largest_x, initial_largest_y, initial_largest_w, initial_largest_h = cv2.boundingRect(contour)

# Draw the largest bounding box in the initial result
cv2.rectangle(initial_result, (initial_largest_x, initial_largest_y), (initial_largest_x + initial_largest_w, initial_largest_y + initial_largest_h), (0, 255, 0), 2)

# Add text to display height and width of the bounding box in the initial result
initial_text = f'Initial BBox Height: {initial_largest_h},\nInitial BBox Width: {initial_largest_w},\nInitial BBox Area: {initial_largest_area}'
cv2.putText(initial_result, initial_text, (initial_largest_x, initial_largest_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the image with the initial bounding box and text using matplotlib
plt.imshow(cv2.cvtColor(initial_result, cv2.COLOR_BGR2RGB))
plt.title('Initial Bounding Box with Dimensions')
plt.axis('off')
plt.show()

# User prompts for input dimensions
provided_height = float(input("Enter provided height in meters: "))
provided_width = float(input("Enter provided width in meters: "))
provided_area = float(input("Enter provided area in square meters:"))

# Calculate the conversion factor based on the initial bounding box and provided dimensions
conversion_factor_area = provided_area / initial_largest_area

# Adjust the initial bounding box based on the single adjustment factor
adjusted_x = initial_largest_x
adjusted_y = initial_largest_y
adjusted_w = initial_largest_w * np.sqrt(conversion_factor_area)
adjusted_h = initial_largest_h * np.sqrt(conversion_factor_area)

# Convert adjusted dimensions back to meters
adjusted_w_meters = adjusted_w / img_width * provided_width
adjusted_h_meters = adjusted_h / img_height * provided_height

# Draw the adjusted bounding box directly on the original image
cv2.rectangle(img, (int(adjusted_x), int(adjusted_y)), (int(adjusted_x + adjusted_w), int(adjusted_y + adjusted_h)), (0, 0, 255), 2)

# Add text to display height and width of the adjusted bounding box in meters
adjusted_text = f'Adjusted BBox Height: {adjusted_h_meters:.3f} meters,\nAdjusted BBox Width: {adjusted_w_meters:.3f} meters,\nAdjusted BBox Area: {adjusted_w_meters * adjusted_h_meters:.3f} square meters'
cv2.putText(img, adjusted_text, (int(adjusted_x), int(adjusted_y + adjusted_h + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display the image with the adjusted bounding box directly on the original image and text using matplotlib
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Adjusted Bounding Box in Original Image with Dimensions in Meters')
plt.axis('off')
plt.show()

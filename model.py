import cv2
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread('examples/test2.jpg')
# Get image dimensions
height, width, _ = image.shape

# Print the dimensions
print(f'Image Width: {width} pixels, Image Height: {height} pixels')
plt.imshow(image)
plt.show()
# Get image dimensions
img_height, img_width, _ = image.shape

# Calculate the dimensions for a bounding box covering 80% of the image
bounding_box_width = int(img_width * 0.85)
bounding_box_height = int(img_height * 0.85)

# Calculate the coordinates for the top-left corner of the bounding box
x = (img_width - bounding_box_width) // 2
y = (img_height - bounding_box_height) // 2

# Create an empty mask
mask = np.zeros((img_height, img_width), dtype=np.uint8)

# Draw the bounding box on the mask
cv2.rectangle(mask, (x, y), (x + bounding_box_width, y + bounding_box_height), 255, thickness=cv2.FILLED)

# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=mask)

# Convert the result to grayscale
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

# Perform edge detection
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
plt.imshow(edged)
plt.show()

# Find contours in the edges
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize variables to store information about the largest bounding box
largest_area = 0
largest_x, largest_y, largest_w, largest_h = 0, 0, 0, 0

# Iterate through contours to find the largest bounding box
for contour in contours:
    area = cv2.contourArea(contour)
    if area > largest_area:
        largest_area = area
        largest_x, largest_y, largest_w, largest_h = cv2.boundingRect(contour)

# Draw the largest bounding box on the image
cv2.rectangle(image, (largest_x, largest_y), (largest_x + largest_w, largest_y + largest_h), (0, 255, 0), 2)

# Add text to display height and width of the bounding box
text = f'Height: {largest_h}, Width: {largest_w}'
cv2.putText(image, text, (largest_x, largest_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with the largest bounding box and text using matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Largest Bounding Box covering 80% of the Image with Dimensions')
plt.axis('off')
plt.show()

# Compare obtained dimensions with provided dimensions
provided_height = 2.972  # meters
provided_width = 1.956  # meters
provided_area = 5.812  # square meters

# Calculate the scale factor based on the provided dimensions and image dimensions
scale_factor_height = provided_height / largest_h
scale_factor_width = provided_width / largest_w
scale_factor_area = provided_area / largest_area

# Calculate the actual dimensions based on the scale factor
actual_height = largest_h * scale_factor_height
actual_width = largest_w * scale_factor_width
actual_area = largest_area * scale_factor_area

# Print the obtained and provided dimensions for comparison
print(f"Obtained Dimensions: Height={actual_height:.3f} meters, Width={actual_width:.3f} meters, Area={actual_area:.3f} square meters")
print(f"Provided Dimensions: Height={provided_height} meters, Width={provided_width} meters, Area={provided_area} square meters")
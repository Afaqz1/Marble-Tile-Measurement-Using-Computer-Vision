from flask import Flask, request, jsonify
import cv2
import numpy as np
from webcolors import rgb_to_name

app = Flask(__name__)

@app.route('/bounding-box', methods=['POST'])
def bounding_box():
    # Get the image file and parameters from the request
    image_file = request.files['image']
    height_meters = float(request.form['height'])
    width_meters = float(request.form['width'])

    # Read the image
    img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ... (rest of the code for processing the image and obtaining the bounding box)

    # Convert pixel dimensions to meters
    width_pixels = largest_w
    height_pixels = largest_h
    pixels_per_meter_width = width_pixels / width_meters
    pixels_per_meter_height = height_pixels / height_meters

    # Convert pixel dimensions to meters
    width_meters = width_pixels / pixels_per_meter_width
    height_meters = height_pixels / pixels_per_meter_height

    # Convert RGB to color name for dominant color
    try:
        dominant_color_name = rgb_to_name(dominant_color)
    except ValueError:
        dominant_color_name = "Unknown Color"

    # Prepare the response
    response = {
        'dominant_color': dominant_color_name,
        'bounding_box': {
            'width': width_meters,
            'height': height_meters,
            'area': largest_area,
            'image_area': image_area,
            'coordinates': {
                'x': largest_x,
                'y': largest_y,
                'w': largest_w,
                'h': largest_h
            }
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
    
"""Standard examples for using the pipeline."""
import sys
import os

# Get the absolute path to the root of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Add the src folder to the Python path
sys.path.insert(0, ROOT_DIR)

import matplotlib

from PyCBD.pipelines import CBDPipeline
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
matplotlib.use('TkAgg')
import cv2


#########
# Basic #
#########

# Load the image.

warped_image_file = r'..\examples\images\broken.jpg'
image = cv2.imread(warped_image_file)

if image is None:
    fallback_path = r'examples\images\broken.jpg'
    image = cv2.imread(fallback_path)

if image is None:
    raise FileNotFoundError("Could not load the image from either path.")

# Create an instance of the detector
detection_pipeline = CBDPipeline()

detection_pipeline.checkerboard_detector.detector.show_processing = False
# Perform detection
# You can optionally give it the checkerboard size, so you'll know whether the coordinates are absolute or relative.
result, board_uv, board_xy = detection_pipeline.detect_checkerboard(image)

#Plot result
fig, ax = plt.subplots()
ax.imshow(image)
ax.plot(board_uv[:, 0], board_uv[:, 1], 'r-o', markeredgecolor='k')
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.4, y=-0.20, units='inches')
ax.text(board_uv[0, 0], board_uv[0, 1], '(' + str(int(board_xy[0, 0])) + ', ' + str(int(board_xy[0, 1])) + ')',
        color="red", transform=trans_offset)
trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.05, units='inches')
ax.text(board_uv[-1, 0], board_uv[-1, 1], '(' + str(int(board_xy[-1, 0])) + ', ' + str(int(board_xy[-1, 1])) + ')',
        color="red", transform=trans_offset)
plt.title("Detection result")
plt.axis('off')
plt.show()


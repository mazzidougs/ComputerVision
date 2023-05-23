# Import the required library
import streamlit as st

# Add a title
st.title("Face Detection using OpenCV " + u' \U0001F468\u200D\U0001F4BB')

# Define the code you want to display
code = """
# import some libraries
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# load the image
img = cv2.imread('IMG_3240.jpeg')  # replace 'path_to_image.jpg' with the actual path to your image

# convert the image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# perform face detection
# to perform better detection you must to adjast the paramaters on the function 'detectMultiScale'
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(30, 30))

# display the image with rectangles and names
fig, ax = plt.subplots(1)
ax.imshow(img_rgb)

for (x, y, w, h) in faces:
    # draw rectangles around the detected faces
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # add names above the rectangles
    ax.text(x, y-10, 'Person', color='g', fontsize=12, fontweight='bold')

# remove axis ticks and labels
ax.axis('off')

# show the image in the notebook
plt.show()
"""

# Display the code using the code method
st.code(code, language='python')

# Define the output of the code (the image)
output_image_path = "/Users/mazzidougs/Documents/Especialização em Computer Vision/Aula 3/img_FaceDetection.png"  # Replace with your image path

# Display the output
st.text("Output:")
st.image(output_image_path)
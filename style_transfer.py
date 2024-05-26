import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
import streamlit as st
import pandas as pd
from io import StringIO

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(img_bytes):
    img = tf.image.decode_image(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]    # Add batch dimension -> [1, height, width, channels]
    return img

# Streamlit UI
st.title('🎨 Neural Style Transfer Web App')
st.write('Upload a photo and select a style to apply neural style transfer.')
st.write('🚀 Tech Stack: Streamlit, TensorFlow')

uploaded_file = st.file_uploader("Choose a content image")
style_option = st.selectbox("Choose a style", ["Van Gogh- Sunflowers", "Van Gogh- The Starry Night", "Frida Kahlo", "Gustav Klimt", "Salvador Dali", "Claude Monet", "Andy Warhol"])

style_images= {
    "Van Gogh- Sunflowers": "image/vangogh.jpeg",
    "Van Gogh- The Starry Night": "image/starrynight.jfif",
    "Frida Kahlo": "image/frida.jpg",
    "Gustav Klimt": "image/klimt.jpg",
    "Salvador Dali": "image/dali.jpeg",
    "Claude Monet": "image/monet.jpeg",
    "Andy Warhol": "image/andy.png"
}

if uploaded_file is not None:
    
    content_image = load_image(uploaded_file.read())
    style_image_path = style_images[style_option]
    # Read the file from style_image_path and returns as a string type
    style_image = load_image(tf.io.read_file(style_image_path))
  
    # st.image function expects a Numpy array
    # [0] indicates the batch dimention -> line 16
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

    st.image(np.squeeze(stylized_image), caption='Stylized Image', use_column_width=True)

    # Convert the tensor to a Numpy array
    stylized_image_np = stylized_image.numpy()

    # Convert the image to the correct format for OpenCV
    stylized_image_cv = cv2.cvtColor(np.squeeze(stylized_image_np) * 255, cv2.COLOR_RGB2BGR)
    
    st.download_button(
        label="Download Stylized Image",
        data=cv2.imencode('.jpg', stylized_image_cv)[1].tobytes(),  # Encode the image into jpg format
        file_name='stylized_image.jpg'
    )
    
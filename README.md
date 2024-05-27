# TensorFlow-neural-style-transfer
## Overview ☁️
This project demonstrates how to use TensorFlow to implement neural style transfer, a technique for blending the artistic style of one image with the content of another. Neural style transfer uses deep learning to combine the content of an image with the style of another, producing a new image that retains the content of the original image while adopting the artistic style of the style image.

## Features 🛠️
* **Content and Style Images**: Use any two images to blend their content and style.
* **Customizable Weights**: Adjust the weights for content and style loss to fine-tune the output.
* **Image Optimization**: Iteratively update the output image to minimize the loss function.
* **TensorFlow Implementation**: Leverages TensorFlow's capabilities for efficient computation and model building.

## Prerequisites ⚙️
```
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
import streamlit as st
```
## How to Run 🧭
```
streamlit run style_transfer.py
```


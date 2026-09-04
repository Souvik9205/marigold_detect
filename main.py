import tensorflow as tf
import numpy as np
import cv2
import sklearn

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("Scikit-learn:", sklearn.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
from google.colab import files
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Load the model
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(7, activation='softmax'))  # Output layer with 7 classes
model.compile('adam', loss=tf.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Define the soil types
soil_types = ['Red Soil', 'Black Soil', 'Clay Soil', 'Sandy Soil', 'Loamy Soil', 'Peaty Soil', 'Chalky Soil']

def suggest_crop(soil_type):
  if soil_type == 'Red Soil':
    return 'Maize, Groundnut, Rice, Mango'
  elif soil_type == 'Black Soil':
    return 'Cotton, Pulses, Millets, Castor, Tobacco, Sugarcane, Citrus fruits, Linseed'
  elif soil_type == 'Clay Soil':
    return 'Rice, Wheat, Barley, Oats'
  elif soil_type == 'Sandy Soil':
    return 'Groundnut, Cotton, Tobacco, Sugarcane'
  elif soil_type == 'Loamy Soil':
    return 'Wheat, Barley, Oats, Maize, Sugarcane'
  elif soil_type == 'Peaty Soil':
    return 'Potatoes, Carrots, Peas, Cucumbers'
  elif soil_type == 'Chalky Soil':
    return 'Wheat, Barley, Oats, Maize'

def suggest_season(crop):
  if crop == 'Maize':
    return 'Kharif season (June to October)'
  elif crop == 'Groundnut':
    return 'Kharif season (June to October)'
  elif crop == 'Rice':
    return 'Kharif season (June to October)'
  elif crop == 'Mango':
    return 'Summer season (March to May)'
  elif crop == 'Cotton':
    return 'Kharif season (June to October)'
  elif crop == 'Pulses':
    return 'Rabi season (October to March)'
  elif crop == 'Millets':
    return 'Kharif season (June to October)'
  elif crop == 'Castor':
    return 'Kharif season (June to October)'
  elif crop == 'Tobacco':
    return 'Rabi season (October to March)'
  elif crop == 'Sugarcane':
    return 'Perennial crop, can be planted throughout the year'
  elif crop == 'Citrus fruits':
    return 'Summer season (March to May)'
  elif crop == 'Linseed':
    return 'Rabi season (October to March)'
  elif crop == 'Wheat':
    return 'Rabi season (October to March)'
  elif crop == 'Barley':
    return 'Rabi season (October to March)'
  elif crop == 'Oats':
    return 'Rabi season (October to March)'
  elif crop == 'Potatoes':
    return 'Rabi season (October to March)'
  elif crop == 'Carrots':
    return 'Rabi season (October to March)'
  elif crop == 'Peas':
    return 'Rabi season (October to March)'
  elif crop == 'Cucumbers':
    return 'Summer season (March to May)'

uploaded = files.upload()

for fn in uploaded.keys():
  # predicting images
  path = '/content/' + fn
  img = cv2.imread(path)
  img = cv2.resize(img, (128, 128))  # Resize to match model input
  img_array = np.array(img)
  img_array = img_array.reshape(1, 128, 128, 3)  # Reshape for model input
  img_array = img_array / 255.0  # Normalize pixel values

  prediction = model.predict(img_array)
  predicted_class = np.argmax(prediction)
  soil_type = soil_types[predicted_class]
  print(f"The image {fn} is likely to be {soil_type}.")

  suggested_crops = suggest_crop(soil_type)
  print(f"The likely crops are: {suggested_crops}")
  for crop in suggested_crops.split(', '):
    season = suggest_season(crop)
    print(f"For {crop}, the suggested season is {season}")

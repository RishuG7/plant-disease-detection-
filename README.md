# Import necessary libraries
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Define the class labels
class_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                'Apple___healthy', 'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
                'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___healthy',
                'Corn___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

# Load and preprocess the input image
image_path = 'test_image.jpg'
image = load_img(image_path, target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image /= 255.

# Make predictions on the input image
predictions = model.predict(image)

# Get the predicted class label and probability
predicted_class_index = np.argmax(predictions[0])
predicted_class_label = class_labels[predicted_class_index]
predicted_probability = predictions[0][predicted_class_index]

# Print the results
print("Predicted class label: {}".format(predicted_class_label))
print("Predicted probability: {:.2%}".format(predicted_probability))

# Plot the input image
plt.imshow(load_img(image_path))
plt.axis('off')
plt.title(predicted_class_label)
plt.show()
# plant-disease-detection-
IoT based projects 

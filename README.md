# Waste Classification Model

## Project Overview
This project focuses on classifying disposed waste into recyclable and non-recyclable categories using image data. By utilizing various convolutional neural network (CNN) architectures, we aim to create a robust model that can accurately identify waste types from images. The primary objectives are to:

- Preprocess the dataset to prepare it for model training.
- Train and evaluate multiple CNN architectures.
- Identify the model with the highest accuracy and interpret its key findings.
- Save the trained models for future use.

## Dataset
The dataset used for this project consists of images of disposed waste, categorized into two classes:

- **Recyclable**
- **Non-Recyclable**

The dataset was split into training and validation sets to evaluate the performance of the models effectively.

## Key Findings
- The baseline CNN model achieved a training accuracy of approximately 94.5% and a validation accuracy of 92.0%.
- The CNN model with dropout layers achieved a training accuracy of approximately 91.3% and a validation accuracy of 93.5%.
- The CNN model with additional convolutional and dense layers achieved a training accuracy of approximately 97.2% and a validation accuracy of 95.0%.

## Instructions for Running the Notebook and Loading the Saved Models

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- The following Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - keras

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras

## Loading the Saved Models
The models have been saved in the models directory. To load a saved model and make predictions, use the following code snippet in a new Jupyter Notebook cell:

from keras.models import load_model

# Load the desired model
model_path = 'saved_models/waste_classification_model.h5'
model = load_model(model_path)

# Example usage: Predicting the class of a new image
import numpy as np
from keras.preprocessing import image

# Load and preprocess the image
img_path = 'data/images/sample_waste.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Make a prediction
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)

# Map the predicted class index to the class label
class_labels = ['Recyclable', 'Non-Recyclable']
print(f'Predicted Class: {class_labels[predicted_class[0]]}')

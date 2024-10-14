# Waste Classification Model

## Project Overview
This project focuses on classifying disposed waste into recyclable and non-recyclable categories using image data. By utilizing various convolutional neural network (CNN) architectures, we aim to create a robust model that can accurately identify waste types from images. The primary objectives are to:

- Preprocess the dataset to prepare it for model training.
- Train and evaluate multiple CNN architectures.
- Identify the model with the highest accuracy and interpret its key findings.
- Save the trained models for future use.

## Dataset
The dataset used for this project is pretty huge(86mb). It consists of images of disposed waste, categorized into 6 classes ( cardboard (393), glass (491), metal (400), paper(584), plastic (472) and trash(127))
Then write a python code to classify cardboard, glass, metal, paper, plastic as  **Recyclable**
And trash as **Non-Recyclable**

The dataset was split into training, testing, and validation sets to evaluate the performance of the models effectively.
### link to Dataset 
(https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data)

## Key Findings
- The vanilla CNN model achieved a training accuracy of approximately 75% and a validation accuracy of 72% and a loss 0.7
- The CNN model with L2 Regularization and Adam optimizer achieved a training accuracy of approximately 69% and a validation accuracy of 59%.
- The CNN model with L2 Regularization and Rmsprop optimizer achieved a training accuracy of approximately 69% and a validation accuracy of 62%.
- The CNN model with L1 Regularization and Adam optimizer achieved a training accuracy of approximately 54% and a validation accuracy of 54%.
- The CNN model with only Adam optimiser and earlystopping(with tweaks in learning rate) achieved a training accuracy of approximately 78% and a validation accuracy of 71.62% and a loss 0.6

## Instructions for Running the Notebook and Loading the Saved Models

### Prerequisites
- Python 3.8 or higher
- Colab
- The following Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - keras

    
Based on the size of the dataset, saved models were pretty huge to upload on github. I will include a link to the model saved on google drive.

( https://drive.google.com/drive/folders/1Huego0_kLFRAwhmSGAWM5im2jYYHO8Rd?usp=drive_link )

You can install the required packages using pip:

```sh
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```
## Loading the Saved Models
The models have been saved in the models directory. To load a saved model and make predictions, use the following code snippet in a new Jupyter Notebook cell:

```sh
from keras.models import load_model
```
### Loading the Model and Making Predictions

To load the trained model and make predictions on new images, you can use the following code:

```sh
from keras.models import load_model

# Load the desired model
model_path = 'saved_models/waste_classification_model.h5'
model = load_model(model_path)

# Mapping from indices to class names
class_names = list(test_generator.class_indices.keys())

# Define recyclable classes
recyclable_classes = {'metal', 'cardboard', 'paper', 'plastic', 'glass'}

# Get the true labels from the test_generator
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Predict using the loaded model
test_generator.reset()
preds = adam_cnn_loaded.predict(test_generator, verbose=1)
predicted_classes = np.argmax(preds, axis=1)

# Function to convert existing classes to "recyclable" or "non-recyclable"
def convert_to_recyclable_non_recyclable(class_name):
    if class_name in recyclable_classes:
        return "recyclable"
    else:
        return "non-recyclable"

# Compare predictions with true labels and visualize
plt.figure(figsize=(20, 10))

for i in range(10):
    # Get the image and its true label
    img, label = test_generator[i]
    true_label = class_labels[np.argmax(label[0])]

    # Convert true label and predicted label to "recyclable" or "non-recyclable"
    true_category = convert_to_recyclable_non_recyclable(true_label)
    predicted_label = class_names[predicted_classes[i]]
    predicted_category = convert_to_recyclable_non_recyclable(predicted_label)

    # Show the image with the predicted and true category
    plt.subplot(2, 5, i + 1)
    plt.imshow(img[0])
    plt.title(f"Pred: {predicted_category}, True: {true_category}")
    plt.axis('off')

plt.show()
# Map the predicted class index to the class label
class_labels = ['Recyclable', 'Non-Recyclable']
print(f'Predicted Class: {class_labels[predicted_class[0]]}')
```

## Optimizations and Parameter Setting: Discussion of Results

### Objective
The primary objective of this project is to accurately classify waste images into two categories: Recyclable and Non-Recyclable. To achieve this, several optimizations and parameter settings were employed to enhance the model's performance.

### Dataset
This is a huge dataset that comprises a total of 2527 images of waste, categorized into 6 classes as mentioned above:


### Model Architecture
The model architecture selected is a Convolutional Neural Network (CNN), which is well-suited for image classification tasks. The initial architecture included:

- Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Dense layers with L1 or L2 for regularization
- Softmax output layer

### Key Findings from Parameter Tuning and Optimizations

1. **Learning Rate Optimization:**
   - **Initial Learning Rate:** Started with a default learning rate of 0.01.
   - **Optimization:** Experimented with different learning rates (0.01, 0.001, 0.0001). The optimal learning rate was found to be 0.0001, which balanced training speed and accuracy without causing the model to overfit.
   - 
2. **Batch Size:**
   - **Initial Batch Size:** Started with a batch size of 32.
   - **Optimization:** A batch size of 32 provided the best balance between computational efficiency and model performance.

3. **Number of Epochs:**
   - **Initial Epochs:** Initially trained for 10 epochs.
   - **Optimization:** Extended to 30 epochs. Early stopping was used to monitor validation loss, which helped in preventing overfitting. The optimal number of epochs was found to be around 30, where the validation loss started to stabilize.

6. **Optimizer Choice:**
   - **Initial Optimizer:** Adam optimizer was used initially due to its adaptive learning rate capabilities.
   - **Optimization:** Compared Adam with RMSprop. Adam remained the best performer in terms of convergence speed and final accuracy.

### Final Model Performance
After implementing the above optimizations and parameter settings, the final model performance improved significantly. The key performance metrics are as follows:

- **Training Accuracy:** 78.0%
- **Validation Accuracy:** 71.6%
- **Test Accuracy:** 72.1%

## Conclusion
The optimizations and parameter tuning significantly enhanced the model's performance, particularly in the training accuracy. Key strategies included adjusting the learning rate, employing appropriate batch sizes, extending the number of epochs with early stopping, utilizing L1 and L2 for regularizations. 

## Later improvement
Due to the size of the dataset(86mb). Training took very long and some regularisation techniques couldn't be applied such as Dropout. Whiles 78% isn't a bad accuracy, it could be improved by using different techniques such as data augmentation.

These steps collectively contributed to achieving a robust and accurate model for waste classification. In future work, I could explore more advanced architectures to further improve accuracy and efficiency.






# ResNet-Emotion: Emotion-Driven Product Image Classification with CAM Visualization

## Overview

This project implements the ResNet-Emotion network, a deep learning model designed for classifying emotion-driven product images and visualizing Class Activation Mapping (CAM) to highlight regions of interest.

## Requirements

Before running the code, make sure you have the following dependencies installed:

- TensorFlow (2.x or later)
- NumPy
- Matplotlib
- OpenCV

To install the required libraries, run:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Dataset Preparation

The model expects two NumPy files:
- `images.npy`: A dataset of images with shape `(N, 256, 256, 3)` where `N` is the number of images, each image is 256x256 pixels with 3 color channels (RGB).
- `labels.npy`: Corresponding labels for the images, with shape `(N,)`. The labels should be integer values, representing the class of each image.

Replace the `images.npy` and `labels.npy` with your dataset.

## Data Preprocessing

- The images are normalized to the range `[0, 1]`.
- Labels are one-hot encoded based on the number of classes.

```python
train_images = train_images.astype('float32') / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
```

## Model Architecture

The ResNet-Emotion network is a Convolutional Neural Network (CNN) based on the ResNet34 architecture. It consists of:
- Convolutional layers with ReLU activation and max pooling.
- Global average pooling followed by fully connected layers for classification.
- The output layer uses softmax activation for multi-class classification.

## Training the Model

The model is trained using the Adam optimizer and categorical crossentropy loss. The dataset is split into training, validation, and test sets (60%/20%/20%).

```python
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

## Model Evaluation

The model is evaluated on the test set for accuracy and loss:

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)
```

## Class Activation Mapping (CAM)

To visualize the important regions of the image that contribute to the model’s prediction, CAM is used. The `plot_cam_and_original` function generates:
- The original image
- The CAM heatmap
- A superimposed image showing the heatmap overlaid on the original image

```python
plot_cam_and_original(model, img, original_img, class_index)
```

## Example Output

The CAM visualization shows:
1. The original image from the test set.
2. The CAM heatmap showing the regions of focus.
3. The superimposed image.

## Conclusion

This project provides an implementation of emotion-driven image classification with CAM visualization. You can easily adapt the code to your dataset and tune the model for different applications in product design and emotion recognition.

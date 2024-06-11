# Tensorflow

## Cats vs. Dogs Classification with TensorFlow

![Cats vs Dogs](https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg)

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Making Predictions](#making-predictions)
- [Saving the Model](#saving-the-model)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Overview

This project showcases a classification model that predicts whether an input image is of a cat or a dog. It was developed for the finale of the Machine Learning and TensorFlow Bootcamp (Skepsis X GDSC SNU).

### Key Objectives:
- Develop a Convolutional Neural Network (CNN) to classify images.
- Preprocess the dataset to improve model performance.
- Implement image augmentation to enhance the model's robustness.
- Evaluate the model and make predictions on custom inputs.

## Project Structure

The repository is organized as follows:

```
Tensorflow
│
├── catvsdog_tensorflow.ipynb     # Notebook for the whole process
└── README.md                     # Readme file
```

Drive Link for the dataset and model : <a href="https://drive.google.com/drive/folders/19_-EqnJwHDi_1hw_kjBZopb1ne1Wex4X?usp=sharing">Dataset_&_Model</a>

Descriptions of the files in drive:

```
Dataset_&_Model
│
├── dogs-vs-cats.zip     # Compressed dataset
└── mymodel1.h5          # Trained model in HDF5 format
```

## Data

The dataset used in this project is the well-known Cats vs. Dogs dataset, sourced from Kaggle. It contains 25,000 images of cats and dogs, split into training and testing datasets.

### Data Preprocessing

- **Splitting**: The dataset is split into training and test sets to evaluate model performance.
- **Scaling**: Image pixel values are scaled to a range of 0 to 1 to facilitate faster training.

## Model Architecture

The model is a Convolutional Neural Network (CNN) designed for image classification tasks. CNNs are highly effective in identifying spatial patterns and features in images.

### Key Layers and Components:

- **Convolutional Layers**: Extract features from input images.
- **Pooling Layers**: Downsample feature maps to reduce dimensionality.
- **Dropout Layers**: Regularize the model to prevent overfitting.
- **Fully Connected Layers**: Perform the final classification.


## Training the Model

The training process involves:
- Loading and augmenting the data.
- Defining the CNN architecture.
- Compiling the model with an optimizer and a loss function.
- Training the model over multiple epochs.

## Evaluating the Model

The model's performance is evaluated using the test set. Metrics such as accuracy, precision, recall, and F1 score are calculated to assess how well the model distinguishes between cats and dogs.

## Making Predictions

You can use the trained model to make predictions on custom images.
This will output whether the image is predicted to be a cat or a dog.

## Saving the Model

The trained model is saved in the HDF5 format and can be reloaded for further training or inference.

## Requirements

Key dependencies include:
- TensorFlow
- Keras
- NumPy
- pandas
- scikit-learn
- matplotlib

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/chandreyeeshome/Tensorflow.git
   cd Tensorflow
   ```

2. **Install the dependencies:**

3. **Prepare the data:**
   
   - Unzip `cats_and_dogs.zip` folder from the given drive link.

## Usage

After setting up the environment, you can train the model, evaluate its performance, and make predictions.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please fork the repository and create a pull request. For major changes, open an issue first to discuss what you would like to change.

## Contact

For any questions or feedback, feel free to contact me through GitHub or via [email](mailto:chandreyeeshome04@gmail.com).

# Cats vs. Dogs Classification with TensorFlow

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
- [Issues](#issues)
- [License](#license)
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
Tensorflow/
│
├── data/
│   ├── cats_and_dogs.zip     # Compressed dataset
│
├── models/
│   ├── cat_dog_classifier.h5 # Trained model in HDF5 format
│
├── notebooks/
│   ├── CatsVsDogs.ipynb      # Jupyter notebook with the entire workflow
│
├── src/
│   ├── data_loader.py        # Script to load and preprocess the dataset
│   ├── model.py              # Script defining the CNN model
│   ├── train.py              # Script to train the model
│   ├── evaluate.py           # Script to evaluate the model
│   ├── predict.py            # Script for making predictions with the model
│
├── results/
│   ├── evaluation_report.txt # Text file with evaluation metrics
│   ├── sample_predictions/   # Directory containing sample prediction images
│
├── README.md                 # Readme file
└── requirements.txt          # List of dependencies
```

### Descriptions of Key Files and Directories

- **data/**: Contains the dataset used for training and testing.
- **models/**: Stores the saved model after training.
- **notebooks/**: Jupyter notebook that documents the entire process from data loading to model evaluation.
- **src/**: Scripts for loading data, building the model, training, evaluating, and making predictions.
- **results/**: Contains evaluation reports and sample prediction outputs.
- **requirements.txt**: Lists the Python packages required to run the project.

## Data

The dataset used in this project is the well-known Cats vs. Dogs dataset, sourced from Kaggle. It contains 25,000 images of cats and dogs, split into training and testing datasets.

### Data Preprocessing

- **Splitting**: The dataset is split into training and test sets to evaluate model performance.
- **Scaling**: Image pixel values are scaled to a range of 0 to 1 to facilitate faster training.
- **Augmentation**: Various transformations like rotation, zoom, and flipping are applied to augment the data.

## Model Architecture

The model is a Convolutional Neural Network (CNN) designed for image classification tasks. CNNs are highly effective in identifying spatial patterns and features in images.

### Key Layers and Components:

- **Convolutional Layers**: Extract features from input images.
- **Pooling Layers**: Downsample feature maps to reduce dimensionality.
- **Dropout Layers**: Regularize the model to prevent overfitting.
- **Fully Connected Layers**: Perform the final classification.

The detailed architecture is defined in the `src/model.py` script and includes several convolutional and pooling layers followed by fully connected layers.

## Training the Model

The training process involves:
- Loading and augmenting the data.
- Defining the CNN architecture.
- Compiling the model with an optimizer and a loss function.
- Training the model over multiple epochs.

### Running the Training Script

To train the model, execute the following command:

```bash
python src/train.py
```

This script will train the model on the augmented dataset and save the trained model in the `models/` directory.

## Evaluating the Model

The model's performance is evaluated using the test set. Metrics such as accuracy, precision, recall, and F1 score are calculated to assess how well the model distinguishes between cats and dogs.

### Running the Evaluation Script

To evaluate the model, use the following command:

```bash
python src/evaluate.py
```

This will generate a report of the model's performance and save it to `results/evaluation_report.txt`.

## Making Predictions

You can use the trained model to make predictions on custom images. The `predict.py` script loads the model and makes predictions on new data.

### Running the Prediction Script

To make predictions on custom inputs, use:

```bash
python src/predict.py --image_path path/to/your/image.jpg
```

This will output whether the image is predicted to be a cat or a dog and display the image with the prediction.

## Saving the Model

The trained model is saved in the HDF5 format and can be reloaded for further training or inference. This is handled within the `train.py` script after the model is trained.

## Requirements

The project requires Python and several libraries, listed in the `requirements.txt` file. Key dependencies include:
- TensorFlow
- Keras
- NumPy
- pandas
- scikit-learn
- matplotlib

### Installing Dependencies

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/chandreyeeshome/Tensorflow.git
   cd Tensorflow
   ```

2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data:**
   
   - Unzip `cats_and_dogs.zip` in the `data/` directory.

## Usage

After setting up the environment, you can train the model, evaluate its performance, and make predictions using the provided scripts. Here's how to get started:

1. **Train the model:**

   ```bash
   python src/train.py
   ```

2. **Evaluate the model:**

   ```bash
   python src/evaluate.py
   ```

3. **Make predictions on custom images:**

   ```bash
   python src/predict.py --image_path path/to/your/image.jpg
   ```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please fork the repository and create a pull request. For major changes, open an issue first to discuss what you would like to change.

## Contact

For any questions or feedback, feel free to contact me through GitHub or via [email](mailto:chandreyeeshome04@gmail.com).

---

You can now copy and paste this formatted `README.md` directly into your repository's README file.

# Neural Network and CNN Image Classification Project

## Introduction

This project focuses on image classification using two different approaches: a basic Neural Network and a Convolutional Neural Network (CNN). The goal is to demonstrate the process of building and training both types of models to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

## Neural Network

### Description

The **Neural Network** part of the project involves implementing a feedforward neural network with the following architecture:

- Input layer: 1024 nodes (after converting images to grayscale and reshaping)
- Hidden layer 1: 16 nodes with sigmoid activation function
- Hidden layer 2: 16 nodes with sigmoid activation function
- Output layer: 4 nodes (one-hot encoded), representing the image categories

### Implementation Details

1. Data Preprocessing:
   - CIFAR-10 images are converted to grayscale to simplify processing.
   - Images are reshaped to 1024 pixels before being fed into the neural network.

2. Training:
   - Batch gradient descent is used for optimization.
   - Backpropagation is implemented to update weights and biases.
   - Training accuracy and loss are recorded for each epoch.

3. Results:
   - Training and validation accuracy are plotted to visualize model performance.
   - The number of correct estimations and accuracy are calculated.

## Convolutional Neural Network (CNN)

### Description

The **Convolutional Neural Network (CNN)** part of the project aims to improve image classification accuracy by utilizing the power of convolutional layers to automatically learn hierarchical features from images.

### CNN Architecture

The CNN architecture comprises the following layers:

1. Convolutional Layer 1:
   - 32 filters with a kernel size of (3, 3).
   - ReLU activation function.

2. Convolutional Layer 2:
   - 64 filters with a kernel size of (3, 3).
   - ReLU activation function.

3. Convolutional Layer 3:
   - 128 filters with a kernel size of (3, 3).
   - ReLU activation function.

4. Max-Pooling Layer:
   - Pooling size of (2, 2).

5. Fully Connected Layer:
   - 128 nodes with ReLU activation function.

6. Output Layer:
   - 10 nodes with softmax activation function for multi-class classification.

### Implementation Details

1. Data Preprocessing:
   - CIFAR-10 images are used as-is (color images).

2. Training:
   - Adam optimizer is used for optimization.
   - Sparse categorical cross-entropy loss function is used for multi-class classification.

3. Results:
   - Training and validation accuracy are plotted to visualize model performance.

## Usage

To run the Neural Network and CNN implementations:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/neural-network-project.git
   cd neural-network-project
   ```

2. Install the required libraries:
   ```
   pip install numpy tensorflow keras matplotlib
   ```

3. Run the Neural Network code:
   ```
   python neural_network.py
   ```

4. Run the CNN code:
   ```
   python cnn.py
   ```

## Conclusion

This project demonstrates the process of building and training both a basic Neural Network and a Convolutional Neural Network for image classification. By comparing the results of both approaches, it provides insights into the benefits of using CNNs for image-related tasks, as well as a foundation for further exploration and experimentation with neural network architectures.

## Contributing, License, and Contact

For information about contributing, license details, and contact information, please refer to the [Contributing](CONTRIBUTING.md), [LICENSE](LICENSE), and [CONTACT](CONTACT.md) files respectively.

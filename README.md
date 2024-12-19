# MNIST Handwritten Digit Recognition with CNN

This project demonstrates a real-time handwritten digit recognition system using a Convolutional Neural Network (CNN). Users can draw digits on a virtual canvas, and the trained model predicts the drawn digit with high accuracy.

---

## Features

- **Real-Time Prediction**: Draw digits directly on a canvas and get instant predictions.
- **Convolutional Neural Network (CNN)**: Utilizes a deep learning model trained on the MNIST dataset for handwritten digit recognition.
- **Data Augmentation**: Incorporates random rotations, zooms, and translations to improve model generalization.
- **Interactive GUI**: Built using `Tkinter` to allow users to interact with the application easily.

---

## Installation and Setup

### Prerequisites

Make sure you have the following installed:
- Python 3.7+
- Required Python libraries: `tensorflow`, `numpy`, `pillow`, `matplotlib`, `opencv-python`, `tkinter`

### Clone the Repository

```bash
git clone https://github.com/AnkithKA/mnist-digit-recognition.git
cd mnist-digit-recognition
### Install Dependencies
```
To install requirements
```bash
pip install -r requirements.txt
```
### Steps to Use the GUI

1. Open the application.
2. Draw a digit (0–9) on the canvas.
3. Click the "Predict" button to see the model's prediction in the console.
4. Click the "Clear" button to reset the canvas.

---

## Model Architecture

- **Input Layer**: Accepts 28x28 grayscale images.
- **Convolutional Layers**:
  - 1st Conv2D: 32 filters, 3x3 kernel
  - 2nd Conv2D: 64 filters, 3x3 kernel
- **Pooling Layers**: MaxPooling2D layers reduce spatial dimensions after each convolution.
- **Fully Connected Layers**:
  - Dense layer with 128 neurons and ReLU activation.
  - Output layer with 10 neurons (softmax activation) for classification.
## Directory Structure
mnist-digit-recognition/  
│   
├── mnist_digit_recognition.py   # Main script for training and prediction   
├── requirements.txt             # List of dependencies   
├── README.md                    # Project documentation   
└── screenshots/                 # Folder to store screenshots   



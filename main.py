import pandas as pandas
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.python.keras import activations
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.metrics import accuracy
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

# Load the MNIST dataset and preprocess
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Data augmentation function
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1)
])

# Enhance the model with CNN layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with augmented data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32).map(lambda x, y: (data_augmentation(x), y))

model.fit(train_dataset, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Model Accuracy: {accuracy}")
print(f"Model Loss: {loss}")

# Real-time digit prediction from drawing
class PaintApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=300, height=300, bg='white')
        self.canvas.pack()

        self.image = Image.new("L", (300, 300), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack()
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()

    def paint(self, event):
        x1, y1 = event.x - 8, event.y - 8
        x2, y2 = event.x + 8, event.y + 8
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (300, 300), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        # Resize image to 28x28 and invert colors
        resized_image = self.image.resize((28, 28), resample=Image.Resampling.LANCZOS)
        inverted_image = ImageOps.invert(resized_image)

        # Convert image to numpy array and normalize
        input_array = np.array(inverted_image) / 255.0
        input_array = input_array.reshape(1, 28, 28, 1)

        # Predict using the model
        prediction = model.predict(input_array)
        digit = np.argmax(prediction)

        print(f"Predicted Digit: {digit}")

# Launch the application
if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()

# Fruit Classifier with Webcam

## Description
This project is a real-time fruit classifier that uses your computer's webcam and a trained Keras deep learning model.  
It detects fruits such as apple, banana, and orange live from the camera feed and displays the predicted fruit name with the confidence score.

The project is implemented using TensorFlow, Keras, OpenCV, and Tkinter for the graphical user interface.

---

## Features
- Real-time video capture from webcam.
- Image preprocessing and normalization.
- Prediction with a pretrained deep learning model.
- Simple GUI to display live video and classification results.



## Installation

Make sure Python 3.7 or higher is installed on your system.

1. Clone the repository:
   ```bash
   git clone https://github.com/Mashael16/fruit_classifier.git
   cd fruit_classifier
(Optional) Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
Install required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Make sure the following files are in the project folder:

keras_model.h5 — your trained Keras model file.

labels.txt — a text file with class names, one per line.

Run the classifier:

bash
Copy code
python classify.py
A GUI window will open displaying the live webcam feed and the predicted fruit with confidence score.

Press the ESC key to exit the program.

Project Files
classify.py: Main Python script that runs the GUI and prediction.

keras_model.h5: Trained Keras model file.

labels.txt: Class labels file.

images/output.png: Screenshot of the GUI output.

requirements.txt: List of Python dependencies.

Dependencies
Python 3.7+

TensorFlow

Keras

OpenCV (opencv-python)

Pillow (PIL)

Tkinter (usually pre-installed with Python)

Notes
Ensure your webcam is properly connected.

Input images are resized to 224x224 pixels before classification.

Images are normalized using (image / 127.5) - 1 for better model performance.

License
This project is licensed under the MIT License.

Contact
For questions or feedback, please contact:
📧 mashael_alharbii@outlook.sa



Thank you for using this fruit classifier! 🍎🍌🍊

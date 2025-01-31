**Face Mask Detection using CNN and DNN** 🧑‍⚖️🤖

This project implements a real-time face mask detection system using Flask, OpenCV, and TensorFlow. It combines Convolutional Neural Networks (CNN) for mask classification and Deep Neural Networks (DNN) for face detection.

**Key Components:**

Face Detection with DNN 👀: The face detection is performed using OpenCV's DNN module, which loads a pre-trained Caffe model (res10_300x300_ssd_iter_140000.caffemodel). It detects faces in video frames captured from the webcam.

_**Face Mask Classification with CNN 😷:**_ After detecting faces, the image regions are passed through a CNN model trained to classify whether the person is wearing a mask or not. The CNN architecture uses convolutional layers for feature extraction and a fully connected layer for classification.

_Model Training 📚:_ The model is trained on a custom dataset containing "with_mask" and "without_mask" categories. The CNN architecture uses ReLU activation in hidden layers and sigmoid activation in the output layer for binary classification. The model is saved as face_mask_detector.h5.

_Real-time Detection ⏱️:_ The system detects faces and predicts the mask status in real-time using webcam input. Detected faces are highlighted with bounding boxes, and the mask status is displayed (green ✅ for mask, red ❌ for no mask).

**Technologies Used:**

- Python 🐍 and TensorFlow for deep learning model development.
- Flask for building the web application.
- OpenCV for face detection and real-time video processing.
- Keras for CNN model architecture.
- NumPy for data manipulation.
- Caffe pre-trained model for face detection via DNN.

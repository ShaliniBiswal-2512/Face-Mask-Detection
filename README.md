Face Mask Detection using CNN and DNN

This project implements a real-time face mask detection system using deep learning techniques. The solution combines Convolutional Neural Networks (CNN) for face mask classification with Deep Neural Networks (DNN) for face detection.

Key Components:

Face Detection with DNN: The face detection is achieved using OpenCV's DNN module, which loads a pre-trained model based on the Caffe framework. This model detects faces in video frames with high accuracy and identifies regions of interest (ROIs) where the faces are located.

Face Mask Classification with CNN: After detecting faces, the regions are passed through a CNN model trained to classify whether the person is wearing a mask or not. The CNN architecture consists of several convolutional layers to extract features from the image and fully connected layers for classification.

Model Training: The CNN model is trained on a custom dataset containing images labeled with "with_mask" and "without_mask" categories. The model uses ReLU activation functions in the hidden layers and Softmax in the output layer for multi-class classification. The model is trained with data augmentation to increase its robustness against variations in images.

Real-time Detection: The system can detect faces and predict whether the individual is wearing a mask in real-time using webcam input. The detected faces are highlighted with bounding boxes, and the mask status is displayed with green (mask) or red (no mask).

Technologies Used:

1. Python and TensorFlow for deep learning model development.
2. OpenCV for face detection and real-time video processing.
3. Keras for CNN model architecture.
4. NumPy and Pandas for data manipulation and preprocessing.
5. Caffe pre-trained model for face detection via DNN.

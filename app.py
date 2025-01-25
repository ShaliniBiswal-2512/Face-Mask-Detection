from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model and handle potential loading errors
try:
    model = tf.keras.models.load_model("face_mask_detector.h5")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None  # Set to None if the model fails to load

categories = ["with_mask", "without_mask"]

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam (camera 0)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("Error: Failed to load the Haar cascade classifier.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            # Extract the face region
            face = frame[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (128, 128)) / 255.0  # Normalize the pixel values
            reshaped_face = np.reshape(resized_face, (1, 128, 128, 3))

            # Make a prediction if the model is loaded
            if model:
                prediction = model.predict(reshaped_face, verbose=0)  # Suppress verbose output
                label = np.argmax(prediction)
                color = (0, 255, 0) if label == 0 else (0, 0, 255)
                label_text = categories[label]  # Remove accuracy info
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                # Default behavior if the model is not loaded
                color = (0, 0, 255)
                cv2.putText(frame, "Model not loaded", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Encode the frame to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            break
        
        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    # Render the main HTML page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Stream video frames to the browser
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Run the Flask application
    app.run(debug=True, host='127.0.0.1', port=5000)
